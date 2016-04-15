/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/
#include <mitsuba/core/fresolver.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/hw/basicshader.h>
#include "microfacet.h"
#include "ior.h"

#define DJ_BRDF_IMPLEMENTATION 1
#include "dj_brdf.h"

MTS_NAMESPACE_BEGIN

/*!\plugin{roughconductor}{Rough conductor material}
 * \order{7}
 * \icon{bsdf_roughconductor}
 * \parameters{
 *     \parameter{distribution}{\String}{
 *          Specifies the type of microfacet normal distribution
 *          used to model the surface roughness.
 *          \vspace{-1mm}
 *       \begin{enumerate}[(i)]
 *           \item \code{beckmann}: Physically-based distribution derived from
 *               Gaussian random surfaces. This is the default.\vspace{-1.5mm}
 *           \item \code{ggx}: The GGX \cite{Walter07Microfacet} distribution (also known as
 *               Trowbridge-Reitz \cite{Trowbridge19975Average} distribution)
 *               was designed to better approximate the long tails observed in measurements
 *               of ground surfaces, which are not modeled by the Beckmann distribution.
 *           \vspace{-1.5mm}
 *           \item \code{phong}: Anisotropic Phong distribution by
 *              Ashikhmin and Shirley \cite{Ashikhmin2005Anisotropic}.
 *              In most cases, the \code{ggx} and \code{beckmann} distributions
 *              should be preferred, since they provide better importance sampling
 *              and accurate shadowing/masking computations.
 *              \vspace{-4mm}
 *       \end{enumerate}
 *     }
 *     \parameter{alpha, alpha1, alpha2}{\Float\Or\Texture}{
 *         Specifies the roughness of the unresolved surface micro-geometry
 *         along the tangent and bitangent directions. When the Beckmann
 *         distribution is used, this parameter is equal to the
 *         \emph{root mean square} (RMS) slope of the microfacets.
 *         \code{alpha} is a convenience parameter to initialize both
 *         \code{alpha1} and \code{alpha2} to the same value. \default{0.1}.
 *     }
 *     \parameter{material}{\String}{Name of a material preset, see
 *           \tblref{conductor-iors}.\!\default{\texttt{Cu} / copper}}
 *     \parameter{eta, k}{\Spectrum}{Real and imaginary components of the material's index of
 *             refraction \default{based on the value of \texttt{material}}}
 *     \parameter{extEta}{\Float\Or\String}{
 *           Real-valued index of refraction of the surrounding dielectric,
 *           or a material name of a dielectric \default{\code{air}}
 *     }
 *     \parameter{specular\showbreak Reflectance}{\Spectrum\Or\Texture}{Optional
 *         factor that can be used to modulate the specular reflection component. Note
 *         that for physical realism, this parameter should never be touched. \default{1.0}}
 * }
 * \vspace{3mm}
 * This plugin implements a realistic microfacet scattering model for rendering
 * rough conducting materials, such as metals. It can be interpreted as a fancy
 * version of the Cook-Torrance model and should be preferred over
 * heuristic models like \pluginref{phong} and \pluginref{ward} if possible.
 * \renderings{
 *     \rendering{Rough copper (Beckmann, $\alpha=0.1$)}
 *     	   {bsdf_roughconductor_copper.jpg}
 *     \rendering{Vertically brushed aluminium (Anisotropic Phong,
 *         $\alpha_u=0.05,\ \alpha_v=0.3$), see
 *         \lstref{roughconductor-aluminium}}
 *         {bsdf_roughconductor_anisotropic_aluminium.jpg}
 * }
 *
 * Microfacet theory describes rough surfaces as an arrangement of unresolved
 * and ideally specular facets, whose normal directions are given by a
 * specially chosen \emph{microfacet distribution}. By accounting for shadowing
 * and masking effects between these facets, it is possible to reproduce the
 * important off-specular reflections peaks observed in real-world measurements
 * of such materials.
 *
 * This plugin is essentially the ``roughened'' equivalent of the (smooth) plugin
 * \pluginref{conductor}. For very low values of $\alpha$, the two will
 * be identical, though scenes using this plugin will take longer to render
 * due to the additional computational burden of tracking surface roughness.
 *
 * The implementation is based on the paper ``Microfacet Models
 * for Refraction through Rough Surfaces'' by Walter et al.
 * \cite{Walter07Microfacet}. It supports three different types of microfacet
 * distributions and has a texturable roughness parameter.
 * To facilitate the tedious task of specifying spectrally-varying index of
 * refraction information, this plugin can access a set of measured materials
 * for which visible-spectrum information was publicly available
 * (see \tblref{conductor-iors} for the full list).
 * There is also a special material profile named \code{none}, which disables
 * the computation of Fresnel reflectances and produces an idealized
 * 100% reflecting mirror.
 *
 * When no parameters are given, the plugin activates the default settings,
 * which describe copper with a medium amount of roughness modeled using a
 * Beckmann distribution.
 *
 * To get an intuition about the effect of the surface roughness parameter
 * $\alpha$, consider the following approximate classification: a value of
 * $\alpha=0.001-0.01$ corresponds to a material with slight imperfections
 * on an otherwise smooth surface finish, $\alpha=0.1$ is relatively rough,
 * and $\alpha=0.3-0.7$ is \emph{extremely} rough (e.g. an etched or ground
 * finish). Values significantly above that are probably not too realistic.
 * \vspace{4mm}
 * \begin{xml}[caption={A material definition for brushed aluminium}, label=lst:roughconductor-aluminium]
 * <bsdf type="roughconductor">
 *     <string name="material" value="Al"/>
 *     <string name="distribution" value="phong"/>
 *     <float name="alpha1" value="0.05"/>
 *     <float name="alpha2" value="0.3"/>
 * </bsdf>
 * \end{xml}
 *
 * \subsubsection*{Technical details}
 * All microfacet distributions allow the specification of two distinct
 * roughness values along the tangent and bitangent directions. This can be
 * used to provide a material with a ``brushed'' appearance. The alignment
 * of the anisotropy will follow the UV parameterization of the underlying
 * mesh. This means that such an anisotropic material cannot be applied to
 * triangle meshes that are missing texture coordinates.
 *
 * \label{sec:visiblenormal-sampling}
 * This plugin uses a new importance sampling technique
 * contributed by Eric Heitz and Eugene D'Eon, which restricts the sampling
 * domain to the set of visible (unmasked) microfacet normals.
 *
 * When using this plugin, you should ideally compile Mitsuba with support for
 * spectral rendering to get the most accurate results. While it also works
 * in RGB mode, the computations will be more approximate in nature.
 * Also note that this material is one-sided---that is, observed from the
 * back side, it will be completely black. If this is undesirable,
 * consider using the \pluginref{twosided} BRDF adapter.
 */
class dj_beckmann_conductor : public BSDF {
public:
	dj_beckmann_conductor(const Properties &props) : BSDF(props) {

		ref<FileResolver> fResolver = Thread::getThread()->getFileResolver();

		m_specularReflectance = new ConstantSpectrumTexture(
			props.getSpectrum("specularReflectance", Spectrum(1.0f)));

		/* set up Fresnel */
		m_mitsubaFresnel = props.getBoolean("mitsubaFresnel", false);
		std::string materialName = props.getString("material", m_mitsubaFresnel ? "Cu" : "none");
		Spectrum intEta, intK;
		if (boost::to_lower_copy(materialName) == "none") {
			intEta = Spectrum(0.0f);
			intK = Spectrum(1.0f);
		} else {
			intEta.fromContinuousSpectrum(InterpolatedSpectrum(
				fResolver->resolve("data/ior/" + materialName + ".eta.spd")));
			intK.fromContinuousSpectrum(InterpolatedSpectrum(
				fResolver->resolve("data/ior/" + materialName + ".k.spd")));
		}
		Float extEta = lookupIOR(props, "extEta", "air");
		m_eta = props.getSpectrum("eta", intEta) / extEta;
		m_k   = props.getSpectrum("k", intK) / extEta;

		/* set up the BRDF */
		float baseRoughness = 1.f;
		if (props.hasProperty("merl")) { // fit a MERL BRDF
			djb::merl merl(props.getString("merl", "").c_str());
			djb::tabular *tab = new djb::tabular(merl, 90);
			djb::microfacet::params params =
				djb::tabular::fit_beckmann_parameters(*tab);

			/* save the roughness parameter */
			params.get_ellipse(&baseRoughness, &baseRoughness);
			m_brdf = new djb::beckmann(tab->get_fresnel());
			delete tab;
		} else { // standard BRDF creation
			m_brdf = new djb::beckmann();
		}

		/* set up roughness */
		if (props.hasProperty("alpha")) {
			m_alpha1 =
			m_alpha2 = new ConstantFloatTexture(baseRoughness * props.getFloat("alpha", 1.0f));
			if (props.hasProperty("alpha1")
			   || props.hasProperty("alpha2")
			   || props.hasProperty("alphaAngle"))
				SLog(EError, "Microfacet model: please specify either 'alpha' or 'alpha1'/'alpha2'/'alphaAngle'.");
		} else if (props.hasProperty("alpha1") || props.hasProperty("alpha2")) {
			if (!props.hasProperty("alpha1") || !props.hasProperty("alpha2"))
				SLog(EError, "Microfacet model: both 'alpha1' and 'alpha2' must be specified.");
			if (props.hasProperty("alpha"))
				SLog(EError, "Microfacet model: please specify either 'alpha' or 'alpha1'/'alpha2'.");
			m_alpha1 = new ConstantFloatTexture(baseRoughness * props.getFloat("alpha1", 1.0f));
			m_alpha2 = new ConstantFloatTexture(baseRoughness * props.getFloat("alpha2", 1.0f));
		} else {
			m_alpha1 = m_alpha2 = new ConstantFloatTexture(baseRoughness);
		}
		m_alphaAngle = new ConstantFloatTexture(M_PI / 180.0 * props.getFloat("alphaAngle", 0.0f));

		/* load LEAN maps */
		m_leanFiltering = props.getBoolean("leanFiltering", true);
		m_leanmap1 = new ConstantSpectrumTexture(
			props.getSpectrum("leanmap1", Spectrum(0.0f)));
		m_leanmap2 = new ConstantSpectrumTexture(
			props.getSpectrum("leanmap2", Spectrum(0.0f)));
		m_dmapScale = props.getFloat("dmapscale", 1.0f);
	}

	dj_beckmann_conductor(Stream *stream, InstanceManager *manager)
	 : BSDF(stream, manager) {
		m_alpha1 = static_cast<Texture *>(manager->getInstance(stream));
		m_alpha2 = static_cast<Texture *>(manager->getInstance(stream));
		m_alphaAngle = static_cast<Texture *>(manager->getInstance(stream));
		m_leanmap1 = static_cast<Texture *>(manager->getInstance(stream));
		m_leanmap2 = static_cast<Texture *>(manager->getInstance(stream));
		m_specularReflectance = static_cast<Texture *>(manager->getInstance(stream));
		m_eta = Spectrum(stream);
		m_k = Spectrum(stream);

		configure();
	}

	~dj_beckmann_conductor()
	{
		delete m_brdf;
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		BSDF::serialize(stream, manager);

		manager->serialize(stream, m_alpha1.get());
		manager->serialize(stream, m_alpha2.get());
		manager->serialize(stream, m_alphaAngle.get());
		manager->serialize(stream, m_leanmap1.get());
		manager->serialize(stream, m_leanmap2.get());
		manager->serialize(stream, m_specularReflectance.get());
		m_eta.serialize(stream);
		m_k.serialize(stream);
	}

	void configure() {
		unsigned int extraFlags = 0;
		if (m_alpha1 != m_alpha2)
			extraFlags |= EAnisotropic;

		if (!m_alpha1->isConstant() || !m_alpha2->isConstant() ||
			!m_specularReflectance->isConstant())
			extraFlags |= ESpatiallyVarying;

		m_components.clear();
		m_components.push_back(EGlossyReflection | EFrontSide | extraFlags);

		/* Verify the input parameters and fix them if necessary */
		m_specularReflectance = ensureEnergyConservation(
			m_specularReflectance, "specularReflectance", 1.0f);

		m_usesRayDifferentials =
			m_alpha1->usesRayDifferentials() ||
			m_alpha2->usesRayDifferentials() ||
			m_specularReflectance->usesRayDifferentials();

		BSDF::configure();
	}

	Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		/* Stop if this component was not requested */
		if (measure != ESolidAngle ||
			/*Frame::cosTheta(bRec.wi) <= 0 ||
			Frame::cosTheta(bRec.wo) <= 0 ||*/
			((bRec.component != -1 && bRec.component != 0) ||
			!(bRec.typeMask & EGlossyReflection)))
			return Spectrum(0.0f);

		/* Construct the microfacet distribution parameters matching the
		   roughness values at the current surface position. */
		djb::microfacet::params params = djb::microfacet::params::elliptic(
			m_alpha1->eval(bRec.its).average(),
			m_alpha2->eval(bRec.its).average(),
			m_alphaAngle->eval(bRec.its).average()
		);
		/* Construct the linear representation for Beckmann */
		Float E1, E2, E3, E4, E5, dummy;
		m_leanmap1->eval(bRec.its).toLinearRGB(E1, E2, dummy);
		m_leanmap2->eval(bRec.its).toLinearRGB(E3, E4, E5);
		const Float BIAS = 25.f;
		E1-= BIAS;
		E2-= BIAS;
		E5-= BIAS*BIAS;
		djb::beckmann::lrep lrep1, lrep2;

		if (m_leanFiltering) { // LEAN filtering
			lrep1 = djb::beckmann::lrep(E1, E2, E3, E4, E5);
		} else { // Naive MIP mapping
			lrep1 = djb::beckmann::lrep(E1, E2, E1*E1, E2*E2, E1*E2);
		}
		lrep1*= m_dmapScale;
		djb::beckmann::params_to_lrep(params, &lrep2);
		/* Get final microfacet Parameters */
		djb::beckmann::lrep_to_params(lrep1 + lrep2, &params);

		/* evaluate the BRDF */
		djb::vec3 o(bRec.wi.x, bRec.wi.y, bRec.wi.z);
		djb::vec3 i(bRec.wo.x, bRec.wo.y, bRec.wo.z);
		djb::vec3 fr_cos = m_brdf->evalp(i, o, (const void*)&params);

		/* Fresnel factor */
		Vector H = normalize(bRec.wo+bRec.wi);
		const Spectrum F = fresnelConductorExact(dot(bRec.wi, H), m_eta, m_k) *
			m_specularReflectance->eval(bRec.its);

		return F * Color3(fr_cos.x, fr_cos.y, fr_cos.z);
	}

	Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (measure != ESolidAngle ||
			/*Frame::cosTheta(bRec.wi) <= 0 ||
			Frame::cosTheta(bRec.wo) <= 0 ||*/
			((bRec.component != -1 && bRec.component != 0) ||
			!(bRec.typeMask & EGlossyReflection)))
			return 0.0f;

		/* Construct the microfacet distribution parameters matching the
		   roughness values at the current surface position. */
		djb::microfacet::params params = djb::microfacet::params::elliptic(
			m_alpha1->eval(bRec.its).average(),
			m_alpha2->eval(bRec.its).average(),
			m_alphaAngle->eval(bRec.its).average()
		);
		/* Construct the linear representation for Beckmann */
		Float E1, E2, E3, E4, E5, dummy;
		m_leanmap1->eval(bRec.its).toLinearRGB(E1, E2, dummy);
		m_leanmap2->eval(bRec.its).toLinearRGB(E3, E4, E5);
		const Float BIAS = 25.f;
		E1-= BIAS;
		E2-= BIAS;
		E5-= BIAS*BIAS;
		djb::beckmann::lrep lrep1, lrep2;

		if (m_leanFiltering) { // LEAN filtering
			lrep1 = djb::beckmann::lrep(E1, E2, E3, E4, E5);
		} else { // Naive MIP mapping
			lrep1 = djb::beckmann::lrep(E1, E2, E1*E1, E2*E2, E1*E2);
		}
		lrep1*= m_dmapScale;
		djb::beckmann::params_to_lrep(params, &lrep2);
		/* Get final microfacet Parameters */
		djb::beckmann::lrep_to_params(lrep1 + lrep2, &params);
		/* Calculate the PDF */
		djb::vec3 o(bRec.wi.x, bRec.wi.y, bRec.wi.z);
		djb::vec3 i(bRec.wo.x, bRec.wo.y, bRec.wo.z);
		float pdf = m_brdf->pdf(i, o, (const void *)&params);

		return pdf;
	}

	Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &sample) const {
		if (/*Frame::cosTheta(bRec.wi) < 0 ||*/
			((bRec.component != -1 && bRec.component != 0) ||
			!(bRec.typeMask & EGlossyReflection)))
			return Spectrum(0.0f);

		/* Construct the microfacet distribution parameters matching the
		   roughness values at the current surface position. */
		djb::microfacet::params params = djb::microfacet::params::elliptic(
			m_alpha1->eval(bRec.its).average(),
			m_alpha2->eval(bRec.its).average(),
			m_alphaAngle->eval(bRec.its).average()
		);
		/* Construct the linear representation for Beckmann */
		Float E1, E2, E3, E4, E5, dummy;
		m_leanmap1->eval(bRec.its).toLinearRGB(E1, E2, dummy);
		m_leanmap2->eval(bRec.its).toLinearRGB(E3, E4, E5);
		const Float BIAS = 25.f;
		E1-= BIAS;
		E2-= BIAS;
		E5-= BIAS*BIAS;
		djb::beckmann::lrep lrep1, lrep2;

		if (m_leanFiltering) { // LEAN filtering
			lrep1 = djb::beckmann::lrep(E1, E2, E3, E4, E5);
		} else { // Naive MIP mapping
			lrep1 = djb::beckmann::lrep(E1, E2, E1*E1, E2*E2, E1*E2);
		}
		lrep1*= m_dmapScale;
		djb::beckmann::params_to_lrep(params, &lrep2);
		/* Get final microfacet Parameters */
		djb::beckmann::lrep_to_params(lrep1 + lrep2, &params);

		/* Importance Sample the BRDF */
		djb::vec3 o(bRec.wi.x, bRec.wi.y, bRec.wi.z);
		djb::vec3 i;
		djb::vec3 fr_cos = m_brdf->evalp_is(
			sample.x,
			sample.y,
			o,
			&i,
			&pdf,
			(const void *)&params
		);

		/* Perfect specular reflection based on the microfacet normal */
		bRec.wo = Normal(i.x, i.y, i.z);
		bRec.eta = 1.0f;
		bRec.sampledComponent = 0;
		bRec.sampledType = EGlossyReflection;

		/* Fresnel factor */
		Vector H = normalize(bRec.wo + bRec.wi);
		const Spectrum F = fresnelConductorExact(dot(bRec.wi, H), m_eta, m_k) *
			m_specularReflectance->eval(bRec.its);

		return F * Color3(fr_cos.x, fr_cos.y, fr_cos.z);
	}

	Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
		Float pdf_ = 0.f;
		return this->sample(bRec, pdf_, sample);
	}

	void addChild(const std::string &name, ConfigurableObject *child) {
		if (child->getClass()->derivesFrom(MTS_CLASS(Texture))) {
			if (name == "alpha")
				m_alpha1 = m_alpha2 = static_cast<Texture *>(child);
			else if (name == "alpha1")
				m_alpha1 = static_cast<Texture *>(child);
			else if (name == "alpha2")
				m_alpha2 = static_cast<Texture *>(child);
			else if (name == "alphaAngle")
				m_alphaAngle = static_cast<Texture *>(child);
			else if (name == "leanmap1")
				m_leanmap1 = static_cast<Texture *>(child);
			else if (name == "leanmap2")
				m_leanmap2 = static_cast<Texture *>(child);
			else if (name == "specularReflectance")
				m_specularReflectance = static_cast<Texture *>(child);
			else
				BSDF::addChild(name, child);
		} else {
			BSDF::addChild(name, child);
		}
	}

	Float getRoughness(const Intersection &its, int component) const {
		return 0.5f * (m_alpha1->eval(its).average()
			+ m_alpha2->eval(its).average());
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "dj_beckmann_conductor[" << endl
			<< "  id = \"" << getID() << "\"," << endl
			<< "  alpha1 = " << indent(m_alpha1->toString()) << "," << endl
			<< "  alpha2 = " << indent(m_alpha2->toString()) << "," << endl
			<< "  alphaAngle = " << indent(m_alphaAngle->toString()) << "," << endl
			<< "  leanmap1 = " << indent(m_leanmap1->toString()) << "," << endl
			<< "  leanmap2 = " << indent(m_leanmap2->toString()) << "," << endl
			<< "  specularReflectance = " << indent(m_specularReflectance->toString()) << "," << endl
			<< "  eta = " << m_eta.toString() << "," << endl
			<< "  k = " << m_k.toString() << endl
			<< "]";
		return oss.str();
	}

	Shader *createShader(Renderer *renderer) const;

	MTS_DECLARE_CLASS()
private:
	ref<Texture> m_specularReflectance;
	ref<Texture> m_alpha1, m_alpha2, m_alphaAngle;
	ref<Texture> m_leanmap1, m_leanmap2;
	djb::brdf *m_brdf;
	Spectrum m_eta, m_k;
	Float m_dmapScale;
	bool m_leanFiltering;
	bool m_mitsubaFresnel;
};

/**
 * GLSL port of the rough conductor shader. This version is much more
 * approximate -- it only supports the Ashikhmin-Shirley distribution,
 * does everything in RGB, and it uses the Schlick approximation to the
 * Fresnel reflectance of conductors. When the roughness is lower than
 * \alpha < 0.2, the shader clamps it to 0.2 so that it will still perform
 * reasonably well in a VPL-based preview.
 */
class dj_beckmann_conductor_shader : public Shader {
public:
	dj_beckmann_conductor_shader(Renderer *renderer, const Texture *specularReflectance,
			const Texture *alpha1, const Texture *alpha2, const Spectrum &eta,
			const Spectrum &k) : Shader(renderer, EBSDFShader),
			m_specularReflectance(specularReflectance), m_alpha1(alpha1), m_alpha2(alpha2){
		m_specularReflectanceShader = renderer->registerShaderForResource(m_specularReflectance.get());
		m_alpha1Shader = renderer->registerShaderForResource(m_alpha1.get());
		m_alpha2Shader = renderer->registerShaderForResource(m_alpha2.get());

		/* Compute the reflectance at perpendicular incidence */
		m_R0 = fresnelConductorExact(1.0f, eta, k);
	}

	bool isComplete() const {
		return m_specularReflectanceShader.get() != NULL &&
			   m_alpha1Shader.get() != NULL &&
			   m_alpha2Shader.get() != NULL;
	}

	void putDependencies(std::vector<Shader *> &deps) {
		deps.push_back(m_specularReflectanceShader.get());
		deps.push_back(m_alpha1Shader.get());
		deps.push_back(m_alpha2Shader.get());
	}

	void cleanup(Renderer *renderer) {
		renderer->unregisterShaderForResource(m_specularReflectance.get());
		renderer->unregisterShaderForResource(m_alpha1.get());
		renderer->unregisterShaderForResource(m_alpha2.get());
	}

	void resolve(const GPUProgram *program, const std::string &evalName, std::vector<int> &parameterIDs) const {
		parameterIDs.push_back(program->getParameterID(evalName + "_R0", false));
	}

	void bind(GPUProgram *program, const std::vector<int> &parameterIDs, int &textureUnitOffset) const {
		program->setParameter(parameterIDs[0], m_R0);
	}

	void generateCode(std::ostringstream &oss,
			const std::string &evalName,
			const std::vector<std::string> &depNames) const {
		oss << "uniform vec3 " << evalName << "_R0;" << endl
			<< endl
			<< "float " << evalName << "_D(vec3 m, float alpha1, float alpha2) {" << endl
			<< "    float ct = cosTheta(m), ds = 1-ct*ct;" << endl
			<< "    if (ds <= 0.0)" << endl
			<< "        return 0.0f;" << endl
			<< "    alpha1 = 2 / (alpha1 * alpha1) - 2;" << endl
			<< "    alpha2 = 2 / (alpha2 * alpha2) - 2;" << endl
			<< "    float exponent = (alpha1*m.x*m.x + alpha2*m.y*m.y)/ds;" << endl
			<< "    return sqrt((alpha1+2) * (alpha2+2)) * 0.15915 * pow(ct, exponent);" << endl
			<< "}" << endl
			<< endl
			<< "float " << evalName << "_G(vec3 m, vec3 wi, vec3 wo) {" << endl
			<< "    if ((dot(wi, m) * cosTheta(wi)) <= 0 || " << endl
			<< "        (dot(wo, m) * cosTheta(wo)) <= 0)" << endl
			<< "        return 0.0;" << endl
			<< "    float nDotM = cosTheta(m);" << endl
			<< "    return min(1.0, min(" << endl
			<< "        abs(2 * nDotM * cosTheta(wo) / dot(wo, m))," << endl
			<< "        abs(2 * nDotM * cosTheta(wi) / dot(wi, m))));" << endl
			<< "}" << endl
			<< endl
			<< "vec3 " << evalName << "_schlick(float ct) {" << endl
			<< "    float ctSqr = ct*ct, ct5 = ctSqr*ctSqr*ct;" << endl
			<< "    return " << evalName << "_R0 + (vec3(1.0) - " << evalName << "_R0) * ct5;" << endl
			<< "}" << endl
			<< endl
			<< "vec3 " << evalName << "(vec2 uv, vec3 wi, vec3 wo) {" << endl
			<< "   if (cosTheta(wi) <= 0 || cosTheta(wo) <= 0)" << endl
			<< "    	return vec3(0.0);" << endl
			<< "   vec3 H = normalize(wi + wo);" << endl
			<< "   vec3 reflectance = " << depNames[0] << "(uv);" << endl
			<< "   float alpha1 = max(0.2, " << depNames[1] << "(uv).r);" << endl
			<< "   float alpha2 = max(0.2, " << depNames[2] << "(uv).r);" << endl
			<< "   float D = " << evalName << "_D(H, alpha1, alpha2)" << ";" << endl
			<< "   float G = " << evalName << "_G(H, wi, wo);" << endl
			<< "   vec3 F = " << evalName << "_schlick(1-dot(wi, H));" << endl
			<< "   return reflectance * F * (D * G / (4*cosTheta(wi)));" << endl
			<< "}" << endl
			<< endl
			<< "vec3 " << evalName << "_diffuse(vec2 uv, vec3 wi, vec3 wo) {" << endl
			<< "    if (cosTheta(wi) < 0.0 || cosTheta(wo) < 0.0)" << endl
			<< "    	return vec3(0.0);" << endl
			<< "    return " << evalName << "_R0 * inv_pi * inv_pi * cosTheta(wo);"<< endl
			<< "}" << endl;
	}
	MTS_DECLARE_CLASS()
private:
	ref<const Texture> m_specularReflectance;
	ref<const Texture> m_alpha1;
	ref<const Texture> m_alpha2;
	ref<Shader> m_specularReflectanceShader;
	ref<Shader> m_alpha1Shader;
	ref<Shader> m_alpha2Shader;
	Spectrum m_R0;
};

Shader *dj_beckmann_conductor::createShader(Renderer *renderer) const {
	return new dj_beckmann_conductor_shader(renderer,
		m_specularReflectance.get(), m_alpha1.get(), m_alpha2.get(), m_eta, m_k);
}

MTS_IMPLEMENT_CLASS(dj_beckmann_conductor_shader, false, Shader)
MTS_IMPLEMENT_CLASS_S(dj_beckmann_conductor, false, BSDF)
MTS_EXPORT_PLUGIN(dj_beckmann_conductor, "Rough conductor BRDF");
MTS_NAMESPACE_END
