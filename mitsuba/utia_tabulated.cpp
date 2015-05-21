

#include <mitsuba/core/fresolver.h>

#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/core/warp.h>

#define DJ_BRDF_IMPLEMENTATION 1
#include "dj_brdf.h"

MTS_NAMESPACE_BEGIN



class GMMBRDF : public BSDF {
public:
	GMMBRDF(const Properties &props)
		: BSDF(props) {

		m_reflectance = new ConstantSpectrumTexture(props.getSpectrum(
			props.hasProperty("reflectance") ? "reflectance"
				: "diffuseReflectance", Spectrum(.5f)));
		
		Float alphaU = props.getFloat("alphaU", 1.0f);
		Float alphaV = props.getFloat("alphaV", 1.0f);
		
		fs::path m_filename = Thread::getThread()->getFileResolver()->resolve(props.getString("filename"));
		
		// load MERL
		djb::utia m_utia(m_filename.string().c_str());
		// load tabulated
		m_tabular_anisotropic = new djb::tabular_anisotropic(djb::microfacet::GAF_SMITH, m_utia, 90, 90, true);

		m_tabular_anisotropic->set_roughness(alphaU, alphaV);
	}

	GMMBRDF(Stream *stream, InstanceManager *manager)
		: BSDF(stream, manager) {

		configure();
	}

	~GMMBRDF()
	{
		delete m_tabular_anisotropic;
	}

	void configure() {
		/* Verify the input parameter and fix them if necessary */
		m_components.clear();
		m_components.push_back(EGlossyReflection | EFrontSide );
		m_usesRayDifferentials = false;
		BSDF::configure();
	}
	
	
	Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (!(bRec.typeMask & EDiffuseReflection) || measure != ESolidAngle
			|| Frame::cosTheta(bRec.wi) <= 0
			|| Frame::cosTheta(bRec.wo) <= 0)
			return Spectrum(0.0f);

		/* Calculate the reflection half-vector */
		Vector H = normalize(bRec.wo+bRec.wi);

		djb::dir wi(djb::vec3(bRec.wi.x, bRec.wi.y, bRec.wi.z));
		djb::dir wo(djb::vec3(bRec.wo.x, bRec.wo.y, bRec.wo.z));
		djb::vec3 value = m_tabular_anisotropic->evalp(wo, wi);

		return Color3(value.x, value.y, value.z);
	}

	Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (!(bRec.typeMask & EDiffuseReflection) || measure != ESolidAngle
			|| Frame::cosTheta(bRec.wi) <= 0
			|| Frame::cosTheta(bRec.wo) <= 0)
			return 0.0f;

		return warp::squareToCosineHemispherePdf(bRec.wo);
	}


	Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
		if (!(bRec.typeMask & EDiffuseReflection) || Frame::cosTheta(bRec.wi) <= 0)
			return Spectrum(0.0f);

		bRec.wo = warp::squareToCosineHemisphere(sample);
		bRec.eta = 1.0f;
		bRec.sampledComponent = 0;
		bRec.sampledType = EDiffuseReflection;

		djb::vec3 wi(bRec.wi.x, bRec.wi.y, bRec.wi.z);
		djb::vec3 wo(bRec.wo.x, bRec.wo.y, bRec.wo.z);
		djb::vec3 brdf = m_tabular_anisotropic->eval(djb::dir(wi),djb::dir(wo));

		return M_PI * Color3(brdf.x, brdf.y, brdf.z);
	}

	Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf_, const Point2 &sample_) const {	
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
		oss << "GMMBRDF[" << endl
			<< "  id = \"" << getID() << "\"," << endl
			<< "]";
		return oss.str();
	}

	Shader *createShader(Renderer *renderer) const;

	MTS_DECLARE_CLASS()
private:
	ref<const Texture> m_reflectance;	

	djb::tabular_anisotropic * m_tabular_anisotropic;
};

// ================ Hardware shader implementation ================

class GMMBRDFShader : public Shader {
public:
	GMMBRDFShader(Renderer *renderer, const Texture *reflectance)
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

Shader *GMMBRDF::createShader(Renderer *renderer) const {
	return new GMMBRDFShader(renderer, m_reflectance.get());
}

MTS_IMPLEMENT_CLASS(GMMBRDFShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(GMMBRDF, false, BSDF)
MTS_EXPORT_PLUGIN(GMMBRDF, "MERL BRDF")
MTS_NAMESPACE_END
