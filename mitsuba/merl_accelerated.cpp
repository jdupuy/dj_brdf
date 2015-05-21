

#include <mitsuba/core/fresolver.h>

#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/core/warp.h>

#define DJ_BRDF_IMPLEMENTATION 1
#include "dj_brdf.h"

#include "microfacet.h"

MTS_NAMESPACE_BEGIN


class MerlBRDF : public BSDF {
public:
	MerlBRDF(const Properties &props)
		: BSDF(props) {

		m_reflectance = new ConstantSpectrumTexture(props.getSpectrum(
			props.hasProperty("reflectance") ? "reflectance"
				: "diffuseReflectance", Spectrum(.5f)));

		fs::path m_filename = Thread::getThread()->getFileResolver()->resolve(props.getString("filename"));

		// load MERL
		m_brdf = new djb::merl(m_filename.string().c_str());
		// load tabulated
		djb::tabular fit(djb::microfacet::GAF_SMITH, *m_brdf, 89, true);
		// load GMM
		m_gmm = new djb::gmm(fit, 8, 100);

		m_Ndistr = m_gmm->get_weights().size();
		m_weight = new float[m_Ndistr];
		m_roughness = new float[m_Ndistr];
		m_distr = new MicrofacetDistribution*[m_Ndistr];
		
		for(int i=0 ; i<m_Ndistr ; ++i)
		{
			m_weight[i] = m_gmm->get_weights()[i];
			double s1, s2, t;
			m_gmm->get_lobes()[i]->get_roughness(&s1, &s2, &t);
			m_roughness[i] = (Float)s1 * sqrtf(2.0);
			m_distr[i] = new MicrofacetDistribution(MicrofacetDistribution::EBeckmann, m_roughness[i], m_roughness[i], false);
		}

	}

	MerlBRDF(Stream *stream, InstanceManager *manager)
		: BSDF(stream, manager) {

		configure();
	}

	~MerlBRDF()
	{
		delete m_brdf;
		delete [] m_weight;
		delete [] m_roughness;
		for(int i=0 ; i<m_Ndistr ; ++i)
			delete m_distr[i];
		delete [] m_distr;
		delete m_gmm;
	}

	void configure() {
		/* Verify the input parameter and fix them if necessary */
		m_components.clear();	
		m_components.push_back(EDiffuseReflection | EFrontSide | 0);
		m_usesRayDifferentials = false;
		BSDF::configure();
	}





	Float D(const Vector& wm) const 
	{
		Float D_ = 0;
		for(int i=0 ; i<m_Ndistr ; ++i)
		{
			D_ += m_weight[i] * m_distr[i]->eval(wm);
		}
		return D_;
	}

	Float PDF(const Vector& wm) const 
	{
		return D(wm) * Frame::cosTheta(wm);
	}

	Float Lambda(const Vector& wi, const Vector &wm) const
	{
		Float Lambda_ = 0;
		for(int i=0 ; i<m_Ndistr ; ++i)
		{
			Lambda_ += m_weight[i] * (1.0f/m_distr[i]->smithG1(wi, wm) - 1.0f);
		}
		return Lambda_;
	}

	Float G1(const Vector& wi, const Vector &wm) const
	{
		if ( dot(wi, wm) * Frame::cosTheta(wi) <= 0 )
			return 0.0f;

		Float Lambdai = Lambda(wi, wm);
		Float G1_ = 1.0f / (1.0f + Lambdai);
		return G1_;
	}

	Float G2(const Vector& wi, const Vector& wo, const Vector &wm) const
	{
		if ( dot(wi, wm) * Frame::cosTheta(wi) <= 0 || dot(wo, wm) * Frame::cosTheta(wo) <= 0 )
			return 0.0f;

		Float Lambdai = Lambda(wi, wm);
		Float Lambdao = Lambda(wo, wm);
		Float G2_ = 1.0f / (1.0f + Lambdai + Lambdao);
		return G2_;
	}



	Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (!(bRec.typeMask & EDiffuseReflection) || measure != ESolidAngle
			|| Frame::cosTheta(bRec.wi) <= 0
			|| Frame::cosTheta(bRec.wo) <= 0)
			return Spectrum(0.0f);

		djb::vec3 wi(bRec.wi.x, bRec.wi.y, bRec.wi.z);
		djb::vec3 wo(bRec.wo.x, bRec.wo.y, bRec.wo.z);
		djb::vec3 brdf = m_brdf->eval(djb::dir(wi),djb::dir(wo));

		return Color3(brdf.x, brdf.y, brdf.z) * Frame::cosTheta(bRec.wo);
	}

	Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (!(bRec.typeMask & EDiffuseReflection) || measure != ESolidAngle
			|| Frame::cosTheta(bRec.wi) <= 0
			|| Frame::cosTheta(bRec.wo) <= 0)
			return 0.0f;

		/* Calculate the reflection half-vector */
		Vector H = normalize(bRec.wo+bRec.wi);

		return PDF(H) / (4 * absDot(bRec.wo, H));
	}


	Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
		if (!(bRec.typeMask & EDiffuseReflection) || Frame::cosTheta(bRec.wi) <= 0)
			return Spectrum(0.0f);

		/* Sample M, the microfacet normal */
		Float pdf_;

		int i=0;
		float sum = m_weight[0];
		float U = (rand()%RAND_MAX) / (Float)RAND_MAX;
		while(sum < U)
		{
			i++;
			sum += m_weight[i];
		}

		Normal m = m_distr[i]->sample(bRec.wi, sample, pdf_);

		if (pdf_ == 0)
			return Spectrum(0.0f);

		/* Perfect specular reflection based on the microfacet normal */
		bRec.wo = reflect(bRec.wi, m);
		bRec.eta = 1.0f;
		bRec.sampledComponent = 0;
		bRec.sampledType = EGlossyReflection;

		/* Side check */
		if (Frame::cosTheta(bRec.wo) <= 0)
			return Spectrum(0.0f);

		pdf_ = pdf(bRec, ESolidAngle);

		// actual weight
		djb::vec3 wi(bRec.wi.x, bRec.wi.y, bRec.wi.z);
		djb::vec3 wo(bRec.wo.x, bRec.wo.y, bRec.wo.z);
		djb::vec3 brdf = m_brdf->eval(djb::dir(wi),djb::dir(wo));

		return Color3(brdf.x, brdf.y, brdf.z) * Frame::cosTheta(bRec.wo) / pdf_;

	}

	Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf_, const Point2 &sample_) const {
		if (!(bRec.typeMask & EDiffuseReflection) || Frame::cosTheta(bRec.wi) <= 0)
			return Spectrum(0.0f);

		Spectrum res = sample(bRec, sample_);
		pdf_ = pdf(bRec, ESolidAngle);
		return res;
	}

	void addChild(const std::string &name, ConfigurableObject *child) {
		if (child->getClass()->derivesFrom(MTS_CLASS(Texture))
				&& (name == "reflectance" || name == "diffuseReflectance")) {

		} else {
			BSDF::addChild(name, child);
		}
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		BSDF::serialize(stream, manager);

	}

	Float getRoughness(const Intersection &its, int component) const {
		return std::numeric_limits<Float>::infinity();
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "MerlBRDF[" << endl
			<< "  id = \"" << getID() << "\"," << endl
			<< "]";
		return oss.str();
	}

	Shader *createShader(Renderer *renderer) const;

	MTS_DECLARE_CLASS()
private:
	ref<const Texture> m_reflectance;
	djb::merl * m_brdf;

	djb::gmm * m_gmm;

	MicrofacetDistribution ** m_distr;
	Float * m_weight;
	Float * m_roughness;
	int m_Ndistr;
};

// ================ Hardware shader implementation ================

class MerlBRDFShader : public Shader {
public:
	MerlBRDFShader(Renderer *renderer, const Texture *reflectance)
		: Shader(renderer, EBSDFShader), m_reflectance(reflectance) {
		m_reflectanceShader = renderer->registerShaderForResource(m_reflectance.get());
	}

	bool isComplete() const {
		return m_reflectanceShader.get() != NULL;
	}

	void cleanup(Renderer *renderer) {
		renderer->unregisterShaderForResource(m_reflectance.get());
	}

	void putDependencies(std::vector<Shader *> &deps) {
		deps.push_back(m_reflectanceShader.get());
	}

	void generateCode(std::ostringstream &oss,
			const std::string &evalName,
			const std::vector<std::string> &depNames) const {
		oss << "vec3 " << evalName << "(vec2 uv, vec3 wi, vec3 wo) {" << endl
			<< "    if (cosTheta(wi) < 0.0 || cosTheta(wo) < 0.0)" << endl
			<< "    	return vec3(0.0);" << endl
			<< "    return " << depNames[0] << "(uv) * inv_pi * cosTheta(wo);" << endl
			<< "}" << endl
			<< endl
			<< "vec3 " << evalName << "_diffuse(vec2 uv, vec3 wi, vec3 wo) {" << endl
			<< "    return " << evalName << "(uv, wi, wo);" << endl
			<< "}" << endl;
	}

	MTS_DECLARE_CLASS()
private:
	ref<const Texture> m_reflectance;
	ref<Shader> m_reflectanceShader;
};

Shader *MerlBRDF::createShader(Renderer *renderer) const {
	return new MerlBRDFShader(renderer, m_reflectance.get());
}

MTS_IMPLEMENT_CLASS(MerlBRDFShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(MerlBRDF, false, BSDF)
MTS_EXPORT_PLUGIN(MerlBRDF, "MERL BRDF")
MTS_NAMESPACE_END
