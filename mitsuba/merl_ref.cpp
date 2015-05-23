

#include <mitsuba/core/fresolver.h>

#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/core/warp.h>

#define DJ_BRDF_IMPLEMENTATION 1
#include "dj_brdf.h"

#include "microfacet.h"

MTS_NAMESPACE_BEGIN


class MERL : public BSDF {
public:
	MERL(const Properties &props)
		: BSDF(props) {

		m_reflectance = new ConstantSpectrumTexture(props.getSpectrum(
			props.hasProperty("reflectance") ? "reflectance"
				: "diffuseReflectance", Spectrum(.5f)));

		fs::path m_filename = Thread::getThread()->getFileResolver()->resolve(props.getString("filename"));

		// load MERL
		m_brdf = new djb::merl(m_filename.string().c_str());
		// load tabulated
		m_tabular = new djb::tabular(djb::microfacet::GAF_SMITH, *m_brdf, 90, true);
	}

	MERL(Stream *stream, InstanceManager *manager)
		: BSDF(stream, manager) {

		configure();
	}

	~MERL()
	{
		delete m_brdf;
		delete m_tabular;
	}

	void configure() {
		/* Verify the input parameter and fix them if necessary */
		m_components.clear();	
		m_components.push_back(EDiffuseReflection | EFrontSide | 0);
		m_usesRayDifferentials = false;
		BSDF::configure();
	}



	Float PDF(const Vector& wm) const 
	{
		djb::dir m(djb::vec3(wm.x, wm.y, wm.z));
		return m_tabular->ndf(m) * Frame::cosTheta(wm);
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
		djb::vec3 wm = djb::vec3(m_tabular->sample(djb::dir(djb::vec3(bRec.wi.x, bRec.wi.y, bRec.wi.z)), sample.x, sample.y));
		
		Normal m(wm.x, wm.y, wm.z);

		Float pdf_ = PDF(m);

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
		oss << "MERL[" << endl
			<< "  id = \"" << getID() << "\"," << endl
			<< "]";
		return oss.str();
	}

	Shader *createShader(Renderer *renderer) const;

	MTS_DECLARE_CLASS()
private:
	ref<const Texture> m_reflectance;
	djb::merl * m_brdf;
	djb::tabular * m_tabular;
};

// ================ Hardware shader implementation ================

class MERLShader : public Shader {
public:
	MERLShader(Renderer *renderer, const Texture *reflectance)
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

Shader *MERL::createShader(Renderer *renderer) const {
	return new MERLShader(renderer, m_reflectance.get());
}

MTS_IMPLEMENT_CLASS(MERLShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(MERL, false, BSDF)
MTS_EXPORT_PLUGIN(MERL, "MERL BRDF")
MTS_NAMESPACE_END
