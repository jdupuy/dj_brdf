

#include <mitsuba/core/fresolver.h>

#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/core/warp.h>

#define DJ_BRDF_IMPLEMENTATION 1
#include "dj_brdf.h"

MTS_NAMESPACE_BEGIN



class MicrofacetGaussian : public BSDF {
public:
	MicrofacetGaussian(const Properties &props)
		: BSDF(props) {

		m_reflectance = new ConstantSpectrumTexture(props.getSpectrum(
			props.hasProperty("reflectance") ? "reflectance"
				: "diffuseReflectance", Spectrum(.5f)));
		
		// roughness parameters
		Float alphaU = props.getFloat("alphaU", 1.0f);
		Float alphaV = props.getFloat("alphaV", 1.0f);
		m_alphaU = new ConstantFloatTexture(alphaU);
		if (alphaU == alphaV)
			m_alphaV = m_alphaU;
		else
			m_alphaV = new ConstantFloatTexture(alphaV);

		// load MERL
		fs::path m_filename = Thread::getThread()->getFileResolver()->resolve(props.getString("filename"));
		djb::merl merl(m_filename.string().c_str());
		// load tabulated
		djb::tabular m_tabular(djb::microfacet::GAF_SMITH, merl, 90, true);
		m_gaussian = djb::tabular::to_gaussian(m_tabular);
		m_gaussian->get_roughness(&reference_fitted_alphaU, &reference_fitted_alphaV, NULL);
	}

	MicrofacetGaussian(Stream *stream, InstanceManager *manager)
		: BSDF(stream, manager) {

		configure();
	}

	~MicrofacetGaussian()
	{
		delete m_gaussian;
	}

	void configure() {
		/* Verify the input parameter and fix them if necessary */
		m_components.clear();
		m_components.push_back(EGlossyReflection | EFrontSide );
		m_usesRayDifferentials = false;
		BSDF::configure();
	}
	
	inline Float projectRoughness(const Vector &v, const Float alphaU, const Float alphaV) const {
		Float invSinTheta2 = 1 / Frame::sinTheta2(v);

		if (invSinTheta2 <= 0)
			return alphaU;

		Float cosPhi2 = v.x * v.x * invSinTheta2;
		Float sinPhi2 = v.y * v.y * invSinTheta2;

		return std::sqrt(cosPhi2 * alphaU * alphaU + sinPhi2 * alphaV * alphaV);
	}
	
	Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (!(bRec.typeMask & EDiffuseReflection) || measure != ESolidAngle
			|| Frame::cosTheta(bRec.wi) <= 0
			|| Frame::cosTheta(bRec.wo) <= 0)
			return Spectrum(0.0f);

		// set roughness
		Float alphaU = m_alphaU->eval(bRec.its).average();
		Float alphaV = m_alphaV->eval(bRec.its).average();
		m_gaussian->set_roughness(alphaU*reference_fitted_alphaU, alphaV*reference_fitted_alphaV);

		djb::dir wi(djb::vec3(bRec.wi.x, bRec.wi.y, bRec.wi.z));
		djb::dir wo(djb::vec3(bRec.wo.x, bRec.wo.y, bRec.wo.z));
		djb::vec3 value = m_gaussian->evalp(wo, wi);

		return Color3(value.x, value.y, value.z);
	}

	Float PDF(const Vector& wm) const 
	{
		djb::dir m(djb::vec3(wm.x, wm.y, wm.z));
		return m_gaussian->ndf(m) * Frame::cosTheta(wm);
	}

	Float G2(const Vector& wi, const Vector& wo, const Vector &wm) const
	{
		if ( dot(wi, wm) * Frame::cosTheta(wi) <= 0 || dot(wo, wm) * Frame::cosTheta(wo) <= 0 )
			return 0.0f;

		djb::dir i(djb::vec3(wi.x, wi.y, wi.z));
		djb::dir o(djb::vec3(wo.x, wo.y, wo.z));
		djb::dir m(djb::vec3(wm.x, wm.y, wm.z));

		return m_gaussian->gaf(i, o, m);
	}

	Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (measure != ESolidAngle ||
			Frame::cosTheta(bRec.wi) <= 0 ||
			Frame::cosTheta(bRec.wo) <= 0 ||
			((bRec.component != -1 && bRec.component != 0) ||
			!(bRec.typeMask & EGlossyReflection)))
			return 0.0f;

		// set roughness
		Float alphaU = m_alphaU->eval(bRec.its).average();
		Float alphaV = m_alphaV->eval(bRec.its).average();
		m_gaussian->set_roughness(alphaU*reference_fitted_alphaU, alphaV*reference_fitted_alphaV);

		/* Calculate the reflection half-vector */
		Vector H = normalize(bRec.wo+bRec.wi);

		return PDF(H) / (4 * absDot(bRec.wo, H));
	}


	Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
		if (Frame::cosTheta(bRec.wi) < 0 ||
			((bRec.component != -1 && bRec.component != 0) ||
			!(bRec.typeMask & EGlossyReflection)))
			return Spectrum(0.0f);
		
		// set roughness
		Float alphaU = m_alphaU->eval(bRec.its).average();
		Float alphaV = m_alphaV->eval(bRec.its).average();
		m_gaussian->set_roughness(alphaU*reference_fitted_alphaU, alphaV*reference_fitted_alphaV);


		djb::vec3 wm = djb::vec3(m_gaussian->sample(djb::dir(djb::vec3(bRec.wi.x, bRec.wi.y, bRec.wi.z)), sample.x, sample.y));
		
		Normal m(wm.x, wm.y, wm.z);
		
		/* Perfect specular reflection based on the microfacet normal */
		bRec.wo = reflect(bRec.wi, m);
		bRec.eta = 1.0f;
		bRec.sampledComponent = 0;
		bRec.sampledType = EGlossyReflection;

		/* Side check */
		if (Frame::cosTheta(bRec.wo) <= 0)
			return Spectrum(0.0f);

		djb::vec3 fresnel = m_gaussian->eval_fresnel(acosf(std::max<Float>(std::min<Float>(1,dot(bRec.wi, m)),0)));
		const Spectrum F = Color3(fresnel.x, fresnel.y, fresnel.z);

		Float weight = G2(bRec.wi, bRec.wo, m) * dot(bRec.wi, m) / (Frame::cosTheta(m) * Frame::cosTheta(bRec.wi));
		
		return F * weight;
	}

	Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf_, const Point2 &sample_) const {	
		Spectrum res = sample(bRec, sample_);
		pdf_ = pdf(bRec, ESolidAngle);
		return res;
	}

	void addChild(const std::string &name, ConfigurableObject *child) {
		if (child->getClass()->derivesFrom(MTS_CLASS(Texture))) {
			if (name == "alpha")
				m_alphaU = m_alphaV = static_cast<Texture *>(child);
			else if (name == "alphaU")
				m_alphaU = static_cast<Texture *>(child);
			else if (name == "alphaV")
				m_alphaV = static_cast<Texture *>(child);
			else
				BSDF::addChild(name, child);
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
		oss << "MicrofacetGaussian[" << endl
			<< "  id = \"" << getID() << "\"," << endl
			<< "]";
		return oss.str();
	}

	Shader *createShader(Renderer *renderer) const;

	MTS_DECLARE_CLASS()
private:
	ref<const Texture> m_reflectance;	
	ref<Texture> m_alphaU, m_alphaV;
	double reference_fitted_alphaU, reference_fitted_alphaV;
	djb::gaussian * m_gaussian;
};

// ================ Hardware shader implementation ================

class MicrofacetGaussianShader : public Shader {
public:
	MicrofacetGaussianShader(Renderer *renderer, const Texture *reflectance)
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

Shader *MicrofacetGaussian::createShader(Renderer *renderer) const {
	return new MicrofacetGaussianShader(renderer, m_reflectance.get());
}

MTS_IMPLEMENT_CLASS(MicrofacetGaussianShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(MicrofacetGaussian, false, BSDF)
MTS_EXPORT_PLUGIN(MicrofacetGaussian, "MERL BRDF")
MTS_NAMESPACE_END
