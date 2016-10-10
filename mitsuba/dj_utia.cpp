

#include <mitsuba/core/fresolver.h>

#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/core/warp.h>
#include <boost/algorithm/string.hpp>


#define DJ_BRDF_IMPLEMENTATION 1
#include "dj_brdf.h"

MTS_NAMESPACE_BEGIN


class dj_utia : public BSDF {
public:
	dj_utia(const Properties &props)
		: BSDF(props) {

		m_reflectance = new ConstantSpectrumTexture(props.getSpectrum(
			props.hasProperty("reflectance") ? "reflectance"
				: "diffuseReflectance", Spectrum(.5f)));

		fs::path m_filename = Thread::getThread()->getFileResolver()->resolve(props.getString("filename"));
		printf("Loading %s\n", m_filename.string().c_str());

		// Retrieve the parameters if specified		
		Float step_t = props.getFloat("step_t", DJB__UTIA_STEP_T),
			step_p = props.getFloat("step_t", DJB__UTIA_STEP_T);
		
		int nti = props.getInteger("nti", DJB__UTIA_NTI),
			ntv = props.getInteger("ntv", DJB__UTIA_NTV);

		djb::utia::FileFormat fileFormat = djb::utia::FileFormat_BIN;
		
        std::string typeF =
            boost::to_lower_copy(props.getString("file_format", "binary"));
        if (typeF == "binary")
            fileFormat = djb::utia::FileFormat_BIN;
        else if (typeF == "exr")
			fileFormat = djb::utia::FileFormat_EXR;
        else if (typeF == "png")
			fileFormat = djb::utia::FileFormat_PNG;
        else
            Log(EError, "Specified an invalid file format type \"%s\", must be "
                "\"binary\", \"exr\", or \"png\"!", typeF.c_str());

		djb::utia::ColorFormat colorFormat = djb::utia::ColorFormat_sRGB;
        std::string typeC =
            boost::to_lower_copy(props.getString("color_format", "srgb"));
        if (typeC == "srgb")
            colorFormat = djb::utia::ColorFormat_sRGB;
        else if (typeC == "xyz")
			colorFormat = djb::utia::ColorFormat_XYZ;
        else
            Log(EError, "Specified an invalid color format type \"%s\", must be "
                "\"sRGB\", or \"XYZ\"!", typeC.c_str());


		
		m_brdf = new djb::utia(m_filename.string().c_str(), step_t, step_p, nti, ntv, fileFormat, colorFormat);
	}

	dj_utia(Stream *stream, InstanceManager *manager)
		: BSDF(stream, manager) {

		configure();
	}

	~dj_utia() {
		delete m_brdf;
	}

	void configure() {
		/* Verify the input parameter and fix them if necessary */
		m_components.clear();	
		m_components.push_back(EDiffuseReflection | EFrontSide | 0);
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
		djb::vec3 fr_p = m_brdf->evalp(i, o);
		return Color3(fr_p.x, fr_p.y, fr_p.z);
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
		djb::vec3 brdf = m_brdf->eval(wi, wo);

		return M_PI * Color3(brdf.x, brdf.y, brdf.z);
	}

	Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &sample) const {
		if (!(bRec.typeMask & EDiffuseReflection) || Frame::cosTheta(bRec.wi) <= 0)
			return Spectrum(0.0f);

		bRec.wo = warp::squareToCosineHemisphere(sample);
		bRec.eta = 1.0f;
		bRec.sampledComponent = 0;
		bRec.sampledType = EDiffuseReflection;
		pdf = warp::squareToCosineHemispherePdf(bRec.wo);

		djb::vec3 wi(bRec.wi.x, bRec.wi.y, bRec.wi.z);
		djb::vec3 wo(bRec.wo.x, bRec.wo.y, bRec.wo.z);
		djb::vec3 brdf = m_brdf->eval(wi, wo);

		return M_PI * Color3(brdf.x, brdf.y, brdf.z);
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
		oss << "dj_utia[" << endl
			<< "  id = \"" << getID() << "\"," << endl
			<< "]";
		return oss.str();
	}

	Shader *createShader(Renderer *renderer) const;

	MTS_DECLARE_CLASS()
private:
	ref<const Texture> m_reflectance;
	djb::utia * m_brdf;
};

// ================ Hardware shader implementation ================

class dj_utia_shader : public Shader {
public:
	dj_utia_shader(Renderer *renderer, const Texture *reflectance)
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

Shader *dj_utia::createShader(Renderer *renderer) const {
	return new dj_utia_shader(renderer, m_reflectance.get());
}

MTS_IMPLEMENT_CLASS(dj_utia_shader, false, Shader)
MTS_IMPLEMENT_CLASS_S(dj_utia, false, BSDF)
MTS_EXPORT_PLUGIN(dj_utia, "dj_utia BRDF")
MTS_NAMESPACE_END
