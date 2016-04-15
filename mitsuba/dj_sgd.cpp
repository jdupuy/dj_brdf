

#include <mitsuba/core/fresolver.h>

#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/core/warp.h>

#define DJ_BRDF_IMPLEMENTATION 1
#include "dj_brdf.h"

#include "microfacet.h"

MTS_NAMESPACE_BEGIN



class dj_sgd : public BSDF {
public:
	dj_sgd(const Properties &props)
		: BSDF(props) {

		m_reflectance = new ConstantSpectrumTexture(props.getSpectrum(
			props.hasProperty("reflectance") ? "reflectance"
				: "diffuseReflectance", Spectrum(.5f)));

		fs::path merlID = Thread::getThread()->getFileResolver()->resolve(props.getString("merlID"));
		m_sgd = new djb::sgd(merlID.string().c_str());
		m_tabular = new djb::tabular(*m_sgd, 90);
	}

	dj_sgd(Stream *stream, InstanceManager *manager)
		: BSDF(stream, manager) {
		m_reflectance = static_cast<Texture *>(manager->getInstance(stream));

		configure();
	}

	~dj_sgd()
	{
		delete m_tabular;
		delete m_sgd;
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

		djb::vec3 o(bRec.wi.x, bRec.wi.y, bRec.wi.z);
		djb::vec3 i(bRec.wo.x, bRec.wo.y, bRec.wo.z);
		djb::vec3 fr_p = m_sgd->evalp(i, o);
		return Color3(fr_p.x, fr_p.y, fr_p.z);
	}

	Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (measure != ESolidAngle ||
			Frame::cosTheta(bRec.wi) <= 0 ||
			Frame::cosTheta(bRec.wo) <= 0 ||
			((bRec.component != -1 && bRec.component != 0) ||
			!(bRec.typeMask & EGlossyReflection)))
			return 0.0f;

		djb::vec3 o(bRec.wi.x, bRec.wi.y, bRec.wi.z);
		djb::vec3 i(bRec.wo.x, bRec.wo.y, bRec.wo.z);
		return m_tabular->pdf(i, o);
	}


	Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
		if (Frame::cosTheta(bRec.wi) < 0 ||
			((bRec.component != -1 && bRec.component != 0) ||
			!(bRec.typeMask & EGlossyReflection)))
			return Spectrum(0.0f);

		/* Sample the tabulated microfacet BRDF */
		djb::vec3 o = djb::vec3(bRec.wi.x, bRec.wi.y, bRec.wi.z);
		djb::vec3 i = m_tabular->sample(sample.x, sample.y, o);

		/* Perfect specular reflection based on the microfacet normal */
		bRec.wo = Vector(i.x, i.y, i.z);
		bRec.eta = 1.0f;
		bRec.sampledComponent = 0;
		bRec.sampledType = EGlossyReflection;

		/* Side check */
		if (Frame::cosTheta(bRec.wo) <= 0)
			return Spectrum(0.0f);

		djb::vec3 fr_p = m_sgd->evalp(i, o) / pdf(bRec, ESolidAngle);
		return Color3(fr_p.x, fr_p.y, fr_p.z);
	}

	Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf_, const Point2 &sample_) const {
		Spectrum res = sample(bRec, sample_);
		pdf_ = pdf(bRec, ESolidAngle);
		return res;
	}

	void addChild(const std::string &name, ConfigurableObject *child) {
		if (child->getClass()->derivesFrom(MTS_CLASS(Texture))
				&& (name == "reflectance" || name == "diffuseReflectance")) {
			m_reflectance = static_cast<Texture *>(child);
		} else {
			BSDF::addChild(name, child);
		}
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		BSDF::serialize(stream, manager);

		manager->serialize(stream, m_reflectance.get());
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "dj_sgd[" << endl
			<< "  id = \"" << getID() << "\"," << endl
			<< "  reflectance = " << indent(m_reflectance->toString()) << endl
			<< "]";
		return oss.str();
	}

	Shader *createShader(Renderer *renderer) const;

	MTS_DECLARE_CLASS()
private:
	ref<const Texture> m_reflectance;
	djb::sgd *m_sgd;
	djb::tabular *m_tabular;
};

// ================ Hardware shader implementation ================

class dj_sgd_shader : public Shader {
public:
	dj_sgd_shader(Renderer *renderer, const Texture *reflectance)
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

Shader *dj_sgd::createShader(Renderer *renderer) const {
	return new dj_sgd_shader(renderer, m_reflectance.get());
}

MTS_IMPLEMENT_CLASS(dj_sgd_shader, false, Shader)
MTS_IMPLEMENT_CLASS_S(dj_sgd, false, BSDF)
MTS_EXPORT_PLUGIN(dj_sgd, "dj_sgd BRDF")
MTS_NAMESPACE_END
