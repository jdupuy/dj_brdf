/* dj_brdf.h - public domain BRDF toolkit
by Jonathan Dupuy

   Do this:
      #define DJ_BRDF_IMPLEMENTATION 1
   before you include this file in *one* C++ file to create the implementation.

   INTERFACING

   define DJB_ASSERT(x) to avoid using assert.h.
   define DJB_LOG(format, ...) to use your own logger (default prints in stdout)

   QUICK NOTES:
       This is research code

*/

/*
TODO
 - implement sampling of visible normals
*/

#ifndef DJB_INCLUDE_DJ_BRDF_H
#define DJB_INCLUDE_DJ_BRDF_H

#include <vector>
#include <string>

namespace djb {

/* Exception API */
struct exc : public std::exception {
	exc(const char *fmt, ...);
	virtual ~exc() throw() {}
	const char *what() const throw() {return m_str.c_str();}
	std::string m_str;
};

/* Utilities */
struct dir;
struct vec3 {
	static vec3 from_raw(const double *v) {return vec3(v[0], v[1], v[2]);}
	static const double *to_raw(const vec3& v) {return &v.x;}
	explicit vec3(double x = 0): x(x), y(x), z(x) {}
	vec3(double x, double y, double z) : x(x), y(y), z(z) {}
	explicit vec3(const dir& dir);
	double intensity() const {return 0.2126 * x + 0.7152 * y + 0.0722 * z;}
	double x, y, z;
};
struct dir {
	explicit dir(double theta = 0, double phi = 0) : theta(theta), phi(phi) {}
	explicit dir(const vec3 &p);
	double theta, phi;
};

/* BRDF interface */
class brdf {
public:
	virtual vec3 eval(const dir& in, const dir& out) const = 0;  // fr
	virtual vec3 eval_hd(const dir& half, const dir& diff) const;
	virtual vec3 evalp(const dir& in, const dir& out) const; // fr * cos
	virtual vec3 evalp_hd(const dir& half, const dir& diff) const;
	virtual vec3 evalp_mc(const dir& out, double u1, double u2, 
	                      dir *in, double *pdf) const; // fr * cos / pdf
	virtual dir sample(const dir& out, double u1, double u2) const;
	virtual double pdf(const dir& out, const dir& m) const;
	virtual ~brdf() {}
	virtual bool is_isotropic() const {return false;}
};

/* MERL BRDF */
class merl : public brdf {
	std::vector<double> m_samples;
public:
	merl(const char *filename);
	vec3 eval(const dir& in, const dir& out) const;
	bool is_isotropic() const {return true;}
	const std::vector<double>& get_samples() const {return m_samples;}
};

/* MIT 2005 BRDF */
class mit05 : public brdf {
	std::vector<float> m_samples;
	int32_t m_dims[4];
public:
	mit05(const char *filename);
	vec3 eval(const dir& in, const dir& out) const;
	const std::vector<float>& get_samples() const {return m_samples;}
private:
	void enforce_reciprocity();
	void dir2index(const dir& in, const dir& out,
	               int *i1, int *i2, int *i3, int *i4) const;
	void index2dir(int i1, int i2, int i3, int i4,
	               dir *in, dir *out) const;
	void rgbindex(int i1, int i2, int i3, int i4,
	              int *r, int *g, int *b) const;
};

/* UTIA BRDF */
class utia : public brdf {
	std::vector<double> m_samples;
	double m_norm;
public:
	utia(const char *filename);
	vec3 eval(const dir& in, const dir& out) const;
	const std::vector<double>& get_samples() const {return m_samples;}
private:
	void normalize();
};

/* Fresnel API */
namespace fresnel {
	class impl {
	public:
		virtual ~impl() {}
		virtual vec3 eval(double theta) const = 0;
		virtual impl *copy() const = 0;
	};

	/* Ideal Specular Reflection */
	class ideal : public impl {
	public:
		vec3 eval(double theta) const {return vec3(1);}
		impl *copy() const {return new ideal();}
	};

	/* Fresnel for Unpolarized Light */
	class unpolarized : public impl {
	public:
		vec3 ior; // index of refraction
		unpolarized(const vec3& ior): ior(ior) {}
		vec3 eval(double theta) const;
		impl *copy() const {return new unpolarized(*this);}
	};

	/* Schlick's Fresnel */
	class schlick : public impl {
	public:
		vec3 f0; // schlick's fresnel factor
		schlick(const vec3& f0): f0(f0) {}
		vec3 eval(double theta) const;
		impl *copy() const {return new schlick(*this);}
	};

	/* SGD's Fresnel */
	class sgd : public impl {
	public:
		vec3 f0, f1;
		sgd(const vec3& f0, const vec3& f1) : f0(f0), f1(f1) {}
		vec3 eval(double theta) const;
		impl *copy() const {return new sgd(*this);}
	};

	/* Arbitrary Fresnel Function */
	class spline : public impl {
		std::vector<vec3> m_points;
	public:
		explicit spline(const std::vector<vec3>& points): m_points(points) {}
		const std::vector<vec3>& get_points() const {return m_points;}
		vec3 eval(double theta) const;
		impl *copy() const {return new spline(*this);}
	};
};

/* Microfacet API */
class microfacet : public brdf {
public:
	enum {GAF_NONE, GAF_VGROOVE, GAF_SMITH};
	virtual ~microfacet() {delete m_fresnel;}
	vec3 eval(const dir& i, const dir& o) const;
	vec3 evalp(const dir& i, const dir& o) const;
	virtual double p22(const dir& h) const = 0;
	virtual double g1(const dir& k) const = 0;
	double ndf(const dir& h) const;
	double gaf(const dir& i, const dir& o, const dir& h) const;
	vec3 eval_fresnel(double theta) const {return m_fresnel->eval(theta);}

	// queries
	virtual bool is_isotropic() const {return m_r1 == m_r2;}

	// mutators
	void set_gaf(int gaf);
	void set_shadow(bool shadow) {m_shadow = shadow;}
	void set_fresnel(const fresnel::impl& f);
	virtual void set_roughness(double r1, double r2, double rangle = 0);
	virtual void set_roughness(double r) {set_roughness(r, r);}
	void set_covariance(double m11, double m22, double m12);

	// accessors
	int get_gaf() const {return m_gaf;}
	int get_shadow() const {return m_shadow;}
	const fresnel::impl& get_fresnel() const {return *m_fresnel;} // FIXME dangerous!
	void get_covariance(double *var1, double *var2, double *cov) const;
	void get_roughness(double *r1, double *r2, double *rangle) const;
protected:
	microfacet(int gaf,
	           double r,
	           const fresnel::impl& f = fresnel::ideal(),
	           bool shadow = true);
	microfacet(int gaf,
	           double r1,
	           double r2,
	           double rangle,
	           const fresnel::impl& f = fresnel::ideal(),
	           bool shadow = true);
	double gaf_nmap(const dir& i, const dir& o, const dir& h) const;
	double gaf_vgroove(const dir& i, const dir& o, const dir& h) const;
	double gaf_smith(const dir& i, const dir& o, const dir& h) const;

	const fresnel::impl *m_fresnel;
	double m_r1, m_r2, m_rangle;
	double m_m11, m_m22, m_m12, m_det;
	double m_sigma1, m_sigma2; // scale
	double m_rho, m_sqrt_one_minus_rho_sqr; // correlation
	int m_gaf;
	bool m_shadow;
};

/* Radial microfacets */
class radial : public microfacet {
public:
	radial(int gaf,
	       double r,
	       const fresnel::impl& fresnel = fresnel::ideal(),
	       bool shadow = true):
		microfacet(gaf, r, fresnel, shadow) {}
	radial(int gaf,
	       double r1,
	       double r2,
	       double rangle,
	       const fresnel::impl& fresnel = fresnel::ideal(),
	       bool shadow = true):
		microfacet(gaf, r1, r2, rangle, fresnel, shadow) {}
	double p22(const dir& h) const;
	double g1(const dir& k) const;
	dir sample(const dir& out, double u1, double u2) const;
	double pdf(const dir& out, const dir& m) const;

	// queries
	virtual double g1_radial(double z) const = 0;
	virtual double p22_radial(double z) const = 0;
	virtual double cdf_radial(double theta) const = 0;
	virtual double qf_radial(double x) const = 0;

	// optional interface with explicit roughness parameters
	// using this interface ensures thread safety for radial microfacet BRDFs
#if 1
	double p22_explicit(const dir& h, double r) const;
	double p22_explicit(const dir& h, double r1, double r2, double rangle = 0.0) const;

	double ndf_explicit(const dir& h, double r) const;
	double ndf_explicit(const dir& h, double r1, double r2, double rangle = 0.0) const;

	double gaf_explicit(const dir& i, const dir& o, const dir& h, double r) const;
	double gaf_explicit(const dir& i, const dir& o, const dir& h, double r1, double r2, double rangle = 0.0) const;

	double g1_explicit(const dir& k, double r) const;
	double g1_explicit(const dir& k, double r1, double r2, double rangle = 0.0) const;

	vec3 evalp_explicit(const dir& i, const dir& o, double r) const;
	vec3 evalp_explicit(const dir& i, const dir& o, double r1, double r2, double rangle = 0.0) const;

	dir sample_explicit(const dir& out, double u1, double u2, double r) const;
	dir sample_explicit(const dir& out, double u1, double u2, double r1, double r2, double rangle = 0.0) const;
#endif
	// end of optional interface

private:
	dir sample_nmap(const dir& out, double u1, double u2) const;
	dir sample_vgroove(const dir& out, double u1, double u2) const;
	dir sample_smith(const dir& out, double u1, double u2) const;
	double pdf_nmap(const dir& out, const dir& m) const;
	double pdf_vgroove(const dir& out, const dir& m) const;
	double pdf_smith(const dir& out, const dir& m) const;
};

/* Gaussian Microfacet NDF */
class gaussian : public radial {
public:
	gaussian(int gaf,
	         double r,
	         const fresnel::impl& fresnel = fresnel::ideal(),
	         bool shadow = true): radial(gaf, r, r, 0, fresnel, shadow)
	{}
	gaussian(int gaf,
	         double r1,
	         double r2,
	         double rangle,
	         const fresnel::impl& fresnel = fresnel::ideal(),
	         bool shadow = true): radial(gaf, r1, r2, rangle, fresnel, shadow)
	{}
	// queries
	double p22_radial(double z) const;
	double g1_radial(double z) const;
	double cdf_radial(double theta) const;
	double qf_radial(double x) const;
};

/* GGX Microfacet NDF */
class ggx : public radial {
public:
	ggx(int gaf,
	    double r,
	    const fresnel::impl& fresnel = fresnel::ideal(),
	    bool shadow = true): radial(gaf, r, r, 0.0, fresnel, shadow)
	{}
	ggx(int gaf,
	    double r1,
	    double r2,
	    double rangle,
	    const fresnel::impl &fresnel = fresnel::ideal(),
	    bool shadow = true): radial(gaf, r1, r2, rangle, fresnel, shadow)
	{}
	// queries
	double p22_radial(double z) const;
	double g1_radial(double z) const;
	double cdf_radial(double theta) const;
	double qf_radial(double x) const;
};

/* Tabulated Microfacet NDF */
class tabular : public radial {
	std::vector<double> m_p22;
	std::vector<double> m_g1;
	std::vector<double> m_cdf;
	std::vector<double> m_qf;
	// tables for VNDF sampling
	std::vector<double> m_pdf1;
	std::vector<double> m_pdf2;
	std::vector<double> m_cdf1;
	std::vector<double> m_cdf2;
	std::vector<double> m_qf1;
	std::vector<double> m_qf2;
public:
	tabular(int gaf, const brdf& brdf, int resolution, bool shadow = true);
	static gaussian *to_gaussian(const tabular& tabular);
	static ggx *to_ggx(const tabular& tabular);
	const std::vector<double>& get_p22v() const {return m_p22;}
	const std::vector<double>& get_g1v() const {return m_g1;}
	const std::vector<double>& get_cdfv() const {return m_cdf;}
	const std::vector<double>& get_qfv() const {return m_qf;}
	// queries
	double p22_radial(double z) const;
	double g1_radial(double z) const;
	double cdf_radial(double theta) const;
	double qf_radial(double x) const;
private:
	void compute_p22_nmap(const brdf& brdf, int res);
	void compute_p22_vgroove(const brdf& brdf, int res);
	void compute_p22_smith(const brdf& brdf, int res);
	void normalize_p22();
	void compute_g1();
	void compute_cdf();
	void compute_qf();
	void compute_fresnel(const brdf& brdf, int res);
};

/* Tabulated Anisotropic Microfacet NDF */
class tabular_anisotropic : public microfacet {
	std::vector<double> m_p22;
	std::vector<double> m_g1;
	std::vector<double> m_pdf1;
	std::vector<double> m_pdf2;
	std::vector<double> m_cdf1;
	std::vector<double> m_cdf2;
	std::vector<double> m_qf1;
	std::vector<double> m_qf2;
	int m_elevation_res;
	int m_azimuthal_res;
public:
	tabular_anisotropic(int gaf,
	                    const brdf& brdf,
	                    int elevation_res,
	                    int azimuthal_res,
	                    bool shadow = true);
	double p22(const dir& h) const;
	double g1(const dir& k) const;
	dir sample(const dir& out, double u1, double u2) const;
	double pdf(const dir& out, const dir& m) const;
	double pdf1(double phi) const;
	double pdf2(double theta, double phi) const;
	double cdf1(double phi) const;
	double cdf2(double theta, double phi) const;
	double qf1(double u1) const;
	double qf2(double u2, double u1) const;
	const std::vector<double>& get_p22v(int *elevc, int *azimc) const;
	const std::vector<double>& get_g1v(int *elevc, int *azimc) const;
	bool is_isotropic() const {return false;}
	static gaussian *to_gaussian(const tabular_anisotropic& tabular);
	static ggx *to_ggx(const tabular_anisotropic& tabular);

	// optional interface with explicit roughness parameters
	// using this interface ensures thread safety for tabular microfacet BRDFs
#if 1
	double p22_explicit(const dir& h, double r) const;
	double p22_explicit(const dir& h, double r1, double r2, double rangle = 0.0) const;

	double ndf_explicit(const dir& h, double r) const;
	double ndf_explicit(const dir& h, double r1, double r2, double rangle = 0.0) const;

	double gaf_explicit(const dir& i, const dir& o, const dir& h, double r) const;
	double gaf_explicit(const dir& i, const dir& o, const dir& h, double r1, double r2, double rangle = 0.0) const;

	double g1_explicit(const dir& k, double r) const;
	double g1_explicit(const dir& k, double r1, double r2, double rangle = 0.0) const;

	vec3 evalp_explicit(const dir& i, const dir& o, double r) const;
	vec3 evalp_explicit(const dir& i, const dir& o, double r1, double r2, double rangle = 0.0) const;

	// XXX not implemented yet
	dir sample_explicit(const dir& out, double u1, double u2, double r) const;
	dir sample_explicit(const dir& out, double u1, double u2, double r1, double r2, double rangle = 0.0) const;
#endif
	// end of optional interface
private:
	double p22(double x, double y) const;
	void normalize_p22();
	void compute_g1();
	void compute_p22_nmap(const brdf& brdf);
	void compute_p22_vgroove(const brdf& brdf);
	void compute_p22_smith(const brdf& brdf);
	void compute_fresnel(const brdf& brdf, int res);
	void compute_pdf1();
	void compute_pdf2();
	void normalize_pdf1();
	void normalize_pdf2();
	void compute_cdf1();
	void compute_cdf2();
	void compute_qf1();
	void compute_qf2();
};

/* Shifted Gamma Distribution BRDF */
class sgd : public brdf {
	struct data {
		const char *name;
		const char *otherName;
		double rhoD[3];
		double rhoS[3];
		double alpha[3];
		double p[3];
		double f0[3];
		double f1[3];
		double kap[3];
		double lambda[3];
		double c[3];
		double k[3];
		double theta0[3];
		double error[3];
	};
	static const data s_data[100];
	const fresnel::impl *m_fresnel;
	const data *m_data;
public:
	sgd(const char *name);
	~sgd() {delete m_fresnel;}
	vec3 eval(const dir& i, const dir& o) const;
	vec3 ndf(const dir& h) const;
	vec3 gaf(const dir& i, const dir& o, const dir& h) const;
	vec3 g1(const dir& k) const;
	vec3 eval_fresnel(double theta) const {return m_fresnel->eval(theta);}
	virtual bool is_isotropic() const {return true;}
	const fresnel::impl &get_fresnel() const {return *m_fresnel;}
};

/* ABC Distribution BRDF */
class abc : public brdf {
	struct data {
		const char *name;
		const char *otherName;
		double kD[3];
		double A[3];
		double B;
		double C;
		double ior;
	};
	static const data s_data[8];
	const fresnel::impl *m_fresnel;
	const data *m_data;
public:
	abc(const char *name);
	~abc() {delete m_fresnel;}
	vec3 eval(const dir& i, const dir& o) const;
	vec3 ndf(const dir& h) const;
	double gaf(const dir& i, const dir& o, const dir& h) const;
	vec3 eval_fresnel(double theta) const {return m_fresnel->eval(theta);}
	virtual bool is_isotropic() const {return true;}
	const fresnel::impl &get_fresnel() const {return *m_fresnel;}
};

} // namespace djb

//
//
//// end header file /////////////////////////////////////////////////////
#endif // DJB_INCLUDE_DJ_BRDF_H

#if DJ_BRDF_IMPLEMENTATION
#include <cmath>
#include <cstdarg>
#include <iostream>     // std::ios, std::istream, std::cout
#include <fstream>      // std::filebuf

#ifndef DJB_ASSERT
#	include <assert.h>
#	define DJB_ASSERT(x) assert(x)
#endif

#ifndef DJB_LOG
#	include <stdio.h>
#	define DJB_LOG(format, ...) fprintf(stdout, format, ##__VA_ARGS__)
#endif

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

namespace djb {

// *************************************************************************************************
// utility API
template<typename T> static T min(const T& a, const T& b) {return a < b ? a : b;}
template<typename T> static T max(const T& a, const T& b) {return a > b ? a : b;}

exc::exc(const char *fmt, ...)
{
	char buf[256];
	va_list args;
	va_start(args, fmt);
	vsnprintf(buf, 256, fmt, args);
	va_end(args);

	m_str = std::string(buf);
}

dir::dir(const vec3& p)
{
	if (p.z > 0.99999) {
		theta = phi = 0.0;
	} else if (p.z < -0.99999) {
		theta = M_PI;
		phi = 0.0;
	} else {
		theta = acos(p.z);
		phi = (p.z != 0.0) ? atan2(p.y, p.x) : 0.0;
	}
}

vec3::vec3(const dir& dir)
{
	double s = sin(dir.theta);
	x = s * cos(dir.phi);
	y = s * sin(dir.phi);
	z = cos(dir.theta);
}

#define OP operator
#define V3 vec3
V3 OP*(double a, const V3& b) {return V3(a * b.x, a * b.y, a * b.z);}
V3 OP*(const V3& a, double b) {return V3(b * a.x, b * a.y, b * a.z);}
V3 OP/(const V3& a, double b) {return (1.0 / b) * a;}
V3 OP*(const V3& a, const V3& b) {return V3(a.x * b.x, a.y * b.y, a.z * b.z);}
V3 OP+(const V3& a, const V3& b) {return V3(a.x + b.x, a.y + b.y, a.z + b.z);}
V3 OP-(const V3& a, const V3& b) {return V3(a.x - b.x, a.y - b.y, a.z - b.z);}
V3& OP+=(V3& a, const V3& b) {a.x+= b.x; a.y+= b.y; a.z+= b.z; return a;}
V3& OP*=(V3& a, const V3& b) {a.x*= b.x; a.y*= b.y; a.z*= b.z; return a;}
V3& OP*=(V3& a, double b) {a.x*= b; a.y*= b; a.z*= b; return a;}
#undef V3
#undef OP

static double dot(const vec3& a, const vec3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

static vec3 cross(const vec3& a, const vec3& b)
{
	return vec3(a.y * b.z - a.z * b.y,
	            a.z * b.x - a.x * b.z,
	            a.x * b.y - a.y * b.x);
}

static vec3 normalize(const vec3& v)
{
	double mag = sqrt(dot(v, v));
	DJB_ASSERT(mag > 0.0 && "invalid vector magnitude");

	return ((1.0 / mag) * v);
}

template<typename T>
static T max3(const T& x, const T& y, const T& z)
{
	T m = x;

	(m < y) && (m = y);
	(m < z) && (m = z);

	return m;
}

static double erf(double x)
{
	// constants
	double a1 =  0.254829592;
	double a2 = -0.284496736;
	double a3 =  1.421413741;
	double a4 = -1.453152027;
	double a5 =  1.061405429;
	double p  =  0.3275911;

	// Save the sign of x
	int sign = 1;
	if (x < 0)
		sign = -1;
	x = fabs(x);

	// A&S formula 7.1.26
	double t = 1.0/(1.0 + p*x);
	double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);

	return sign*y;
}

// *************************************************************************************************
// Geometric operations

//---------------------------------------------------------------------------
// rotate vector along one axis
static vec3 rotate_vector(const vec3& in, const vec3& axis, double angle)
{
	double cos_angle = cos(angle);
	vec3 out = cos_angle * in;
	double tmp1 = dot(axis, in);
	double tmp2 = tmp1 * (1.0 - cos_angle);
	out+= axis * tmp2;
	out+= sin(angle) * cross(axis, in);

	return out;
}

//---------------------------------------------------------------------------
// convert standard coordinates to half vector/difference vector coordinates
static void std_coords_to_half_diff_coords(
	const dir& in, const dir& out,
	dir& half, dir& diff
) {
	vec3 i = vec3(in);
	vec3 o = vec3(out);
	vec3 h = normalize(i + o);
	vec3 t = vec3(0, 1, 0);
	vec3 n = vec3(0, 0, 1);
	half = dir(h);

	vec3 tmp = rotate_vector(i, n, -half.phi);
	vec3 d = normalize(rotate_vector(tmp, t, -half.theta));
	diff = dir(d);
}

//---------------------------------------------------------------------------
// convert half vector/difference vector coordinates to standard coordinates
static void
half_diff_coords_to_std(
	const dir& half, const dir& diff,
	dir& in, dir& out
) {
	vec3 h = vec3(half);
	vec3 d = vec3(diff);
	vec3 n = vec3(0, 0, 1);
	vec3 t = vec3(0, 1, 0);

	vec3 tmp = rotate_vector(d, t, half.theta);
	vec3 i = normalize(rotate_vector(tmp, n, half.phi));
	vec3 o = 2.0 * dot(i, h) * h - i;

	in = dir(in);
	out = dir(o);
}

// *************************************************************************************************
// BRDF API

vec3 brdf::eval_hd(const dir& half, const dir& diff) const
{
	dir in, out;
	half_diff_coords_to_std(half, diff, in, out);

	return eval(in, out);
}

vec3 brdf::evalp(const dir& in, const dir& out) const
{
	return cos(in.theta) * eval(in, out);
}

vec3 brdf::evalp_hd(const dir& half, const dir& diff) const
{
	dir in, out;
	half_diff_coords_to_std(half, diff, in, out);

	return cos(in.theta) * eval(in, out);
}

vec3
brdf::evalp_mc(const dir& out, double r1, double r2, dir *in, double *pdf) const
{
	const dir in_ = sample(out, r1, r2);
	double pdf_ = this->pdf(out, in_);
	if (in) (*in) = in_;
	if (pdf) (*pdf) = pdf_;

	return (evalp(in_, out) / pdf_);
}

dir brdf::sample(const dir& out, double u1, double u2) const
{
	double tmp = sqrt(u1 / (1.0 - u1));
	double theta = atan(tmp);
	double phi = u2 * 2.0 * M_PI;

	return dir(theta, phi);
}

double brdf::pdf(const dir& out, const dir& m) const
{
	return cos(m.theta) / M_PI;
}


// *************************************************************************************************
// MERL API implementation

// Copyright 2005 Mitsubishi Electric Research Laboratories All Rights Reserved.

// Permission to use, copy and modify this software and its documentation without
// fee for educational, research and non-profit purposes, is hereby granted, provided
// that the above copyright notice and the following three paragraphs appear in all copies.

// To request permission to incorporate this software into commercial products contact:
// Vice President of Marketing and Business Development;
// Mitsubishi Electric Research Laboratories (MERL), 201 Broadway, Cambridge, MA 02139 or 
// <license@merl.com>.

// IN NO EVENT SHALL MERL BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL,
// OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND
// ITS DOCUMENTATION, EVEN IF MERL HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

// MERL SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED
// HEREUNDER IS ON AN "AS IS" BASIS, AND MERL HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT,
// UPDATES, ENHANCEMENTS OR MODIFICATIONS.

#define MERL_SAMPLING_RES_THETA_H   90
#define MERL_SAMPLING_RES_THETA_D   90
#define MERL_SAMPLING_RES_PHI_D    360

#define MERL_RED_SCALE   (1.00 / 1500.0)
#define MERL_GREEN_SCALE (1.15 / 1500.0)
#define MERL_BLUE_SCALE  (1.66 / 1500.0)

//---------------------------------------------------------------------------
// Lookup theta_half index
// This is a non-linear mapping!
// In:  [0 .. pi/2]
// Out: [0 .. 89]
static int 
theta_half_index(double theta_half)
{
	if (theta_half <= 0.0)
		return 0;
	double theta_half_deg = ((theta_half / (M_PI/2.0)) * MERL_SAMPLING_RES_THETA_H);
	double temp = theta_half_deg * MERL_SAMPLING_RES_THETA_H;
	temp = sqrt(temp);
	int ret_val = (int)temp;
	if (ret_val < 0)
		ret_val = 0;
	else if (ret_val >= MERL_SAMPLING_RES_THETA_H)
		ret_val = MERL_SAMPLING_RES_THETA_H - 1;
	return ret_val;
}

//---------------------------------------------------------------------------
// Lookup theta_diff index
// In:  [0 .. pi/2]
// Out: [0 .. 89]
static int 
theta_diff_index(double theta_diff)
{
	int tmp = theta_diff / (M_PI * 0.5) * MERL_SAMPLING_RES_THETA_D;
	if (tmp < 0)
		return 0;
	else if (tmp < MERL_SAMPLING_RES_THETA_D - 1)
		return tmp;
	else
		return MERL_SAMPLING_RES_THETA_D - 1;
}

//---------------------------------------------------------------------------
// Lookup phi_diff index
static int
phi_diff_index(double phi_diff)
{
	// Because of reciprocity, the BRDF is unchanged under
	// phi_diff -> phi_diff + M_PI
	if (phi_diff < 0.0)
		phi_diff += M_PI;

	// In: phi_diff in [0 .. pi]
	// Out: tmp in [0 .. 179]
	int tmp = phi_diff / M_PI * MERL_SAMPLING_RES_PHI_D / 2;
	if (tmp < 0)
		return 0;
	else if (tmp < MERL_SAMPLING_RES_PHI_D / 2 - 1)
		return tmp;
	else
		return MERL_SAMPLING_RES_PHI_D / 2 - 1;
}
// XXX End of 
// Copyright 2005 Mitsubishi Electric Research Laboratories All Rights Reserved.

//---------------------------------------------------------------------------
// MERL Contructor
merl::merl(const char *filename)
{
	std::fstream f(filename, std::fstream::in | std::fstream::binary);
	int32_t n, dims[3];

	// check file
	if (!f.is_open())
		throw exc("djb_error: Failed to open %s\n", filename);

	// read header
	f.read((char *)dims, /*bytes*/4 * 3);
	n = dims[0] * dims[1] * dims[2];

	// allocate brdf and read data
	m_samples.resize(3 * n);
	f.read((char *)&m_samples[0], sizeof(double) * 3 * n);
	if (f.fail())
		throw exc("djb_error: Reading %s failed\n", filename);
}

//---------------------------------------------------------------------------
// look up the BRDF.
vec3 merl::eval(const dir& in, const dir& out) const
{
	// convert to half / diff angle coordinates
	dir half, diff;
	std_coords_to_half_diff_coords(in, out, half, diff);

	// compute indexes
	int idx_r = phi_diff_index(diff.phi)
	          + theta_diff_index(diff.theta)
	          * MERL_SAMPLING_RES_PHI_D / 2
	          + theta_half_index(half.theta)
	          * MERL_SAMPLING_RES_PHI_D / 2
	          * MERL_SAMPLING_RES_THETA_D;
	int idx_g = idx_r + MERL_SAMPLING_RES_THETA_H
	          * MERL_SAMPLING_RES_THETA_D 
	          * MERL_SAMPLING_RES_PHI_D / 2;
	int idx_b = idx_r + MERL_SAMPLING_RES_THETA_H
	          * MERL_SAMPLING_RES_THETA_D 
	          * MERL_SAMPLING_RES_PHI_D;

	// get color
	vec3 rgb;
	rgb.x = m_samples[idx_r] * MERL_RED_SCALE;
	rgb.y = m_samples[idx_g] * MERL_GREEN_SCALE;
	rgb.z = m_samples[idx_b] * MERL_BLUE_SCALE;

	if (rgb.x < 0.0 || rgb.y < 0.0 || rgb.z < 0.0) {
#ifndef NVERBOSE
	DJB_LOG("djb_verbose: below horizon\n");
#endif
		return vec3(0, 0, 0);
	}

	return rgb;
}

// *************************************************************************************************
// MIT 2005 API implementation (based on Addy Ngan's Matlab implementation)
//---------------------------------------------------------------------------
// MIT 2005 Contructor
mit05::mit05(const char *filename)
{
	std::fstream f(filename, std::fstream::in | std::fstream::binary);
	int32_t n, head[16];

	// open file
	if (!f.is_open())
		throw exc("djb_error: Failed to open %s\n", filename);

	// read header
	f.read((char *)head, sizeof(head));
	memcpy(m_dims, head, /*bytes*/sizeof(m_dims));
	n = head[0] * head[1] * head[2] * head[3];
	if (n < 1)
		throw exc("djb_error: Invalid MIT05 dimensions\n");
	if (head[9] == 1)
		throw exc("djb_error: No support for halved data\n");

	// allocate and read data
	m_samples.resize(3 * n);
	f.read((char *)&m_samples[0], /*bytes*/sizeof(float) * 3 * n);
	if (f.fail())
		throw exc("djb_error: Reading %s failed\n", filename);

	// post process data
	enforce_reciprocity();

#ifndef NVERBOSE
	DJB_LOG("djb_verbose: BRDF resolution %ix%ix%ix%i\n",
	        m_dims[0], m_dims[1], m_dims[2], m_dims[3]);
#endif
}

//---------------------------------------------------------------------------
// look up the BRDF.
vec3 mit05::eval(const dir& in, const dir& out) const
{
	// check that we're above horizon
	if (in.theta > M_PI * 0.5 || out.theta > M_PI * 0.5)
		return vec3(0, 0, 0);

	int i1, i2, i3, i4;
	int i_r, i_g, i_b;
	dir2index(in, out, &i1, &i2, &i3, &i4);
	rgbindex(i1, i2, i3, i4, &i_r, &i_g, &i_b);

	// lookup data
	vec3 rgb(m_samples[i_r], m_samples[i_g], m_samples[i_b]);
	if (rgb.x < 0.0) rgb.x = 0.0;
	if (rgb.y < 0.0) rgb.y = 0.0;
	if (rgb.z < 0.0) rgb.z = 0.0;

	return (1e2 * rgb);
}

//---------------------------------------------------------------------------
// Indexing utilities
void
mit05::dir2index(
	const dir& in, const dir& out, 
	int *i1, int *i2, int *i3, int *i4
) const {
	// check angles
	const double pi = 3.14159;
	double phi_i = in.phi;
	double phi_d = out.phi - in.phi;
	while (phi_i < 0.0) {phi_i+= 2.0 * pi;}
	while (phi_d < 0.0) {phi_d+= 2.0 * pi;}
	while (phi_i > 2.0 * pi) {phi_i-= 2.0 * pi;}
	while (phi_d > 2.0 * pi) {phi_d-= 2.0 * pi;}

	// compute indexes
	(*i1) = floor(min(1.0, in.theta * 2.0 / pi) * (m_dims[0] - 1) + 0.5);
	(*i2) = floor(min(1.0, out.theta * 2.0 / pi) * (m_dims[1] - 1) + 0.5);
	(*i3) = floor(min(1.0, phi_d * 0.5 / pi) * (m_dims[2] - 1) + 0.5);
	(*i4) = floor(min(1.0, phi_i * 0.5 / pi) * (m_dims[3] - 1) + 0.5);
}

void
mit05::index2dir(
	int i1, int i2, int i3, int i4,
	dir *in, dir *out
) const {
	const double pi = 3.14159;
	double tmp1 = (double)i1 / (double)(m_dims[0] - 1);
	double tmp2 = (double)i2 / (double)(m_dims[1] - 1);
	double tmp3 = (double)i3 / (double)(m_dims[2] - 1);
	double tmp4 = (double)i4 / (double)(m_dims[3] - 1);
	double theta_i = tmp1 * 0.5 * pi;
	double theta_o = tmp2 * 0.5 * pi;
	double phi_d = tmp3 * 2.0 * pi;
	double phi_i = tmp4 * 2.0 * pi;
	double phi_o = phi_d + phi_i;
	if (phi_o > 2.0 * pi) phi_o-= 2.0 * pi;
	(*in) = dir(theta_i, phi_i);
	(*out) = dir(theta_o, phi_o);
}

void
mit05::rgbindex(
	int i1, int i2, int i3, int i4,
	int *r, int *g, int *b
) const {
	int tmp = m_dims[0] * m_dims[1]* m_dims[2] * m_dims[3];
	(*r) = i4 + m_dims[3] * (i3 + m_dims[2] * (i2 + m_dims[1] * i1));
	(*g) = (*r) + tmp;
	(*b) = (*g) + tmp;
}

//---------------------------------------------------------------------------
// Post process the BRDF to make sure it is reciproqual
void mit05::enforce_reciprocity()
{
	DJB_ASSERT(m_dims[0] == m_dims[1]);
#if 1
	for (int i1 = 0; i1 < m_dims[0]; ++i1)
	for (int i2 = 0; i2 < m_dims[1]; ++i2)
	for (int i3 = 0; i3 < m_dims[2]; ++i3)
	for (int i4 = 0; i4 < m_dims[3]; ++i4)
	{
		dir i, o;
		index2dir(i1, i2, i3, i4, &i, &o);
		int j1, j2, j3, j4;
		dir2index(o, i, &j1, &j2, &j3, &j4);
		DJB_ASSERT(i1 == j2);
		int r1, g1, b1;
		int r2, g2, b2;
		rgbindex(i1, i2, i3, i4, &r1, &g1, &b1);
		rgbindex(j1, j2, j3, j4, &r2, &g2, &b2);

		// average data
		double tmp1 = 0.5 * (m_samples[r1] + m_samples[r2]);
		double tmp2 = 0.5 * (m_samples[g1] + m_samples[g2]);
		double tmp3 = 0.5 * (m_samples[b1] + m_samples[b2]);
		m_samples[r1] = m_samples[r2] = tmp1;
		m_samples[g1] = m_samples[g2] = tmp2;
		m_samples[b1] = m_samples[b2] = tmp3;

#if 0 // debug index routines
		{
			int t1, t2, t3, t4;
			dir2index(i, o, &t1, &t2, &t3, &t4);
		//	printf("%i vs %i\n", i1, t1);
		//	printf("%i vs %i\n", i2, t2);
		//	printf("%i vs %i\n", i3, t3);
		//	printf("%i vs %i\n\n", i4, t4);
		//	DJB_ASSERT(i1 == t1);
		//	DJB_ASSERT(i2 == t2);
		//	DJB_ASSERT(i3 == t3);
		//	DJB_ASSERT(i4 == t4);
		}
#endif
	}
#endif
}

// *************************************************************************************************
// UTIA API implementation (based on Jiri Filip's implementation)

#define DJB__UTIA_STEP_T  15.0
#define DJB__UTIA_STEP_P   7.5
#define DJB__UTIA_NTI      6
#define DJB__UTIA_NPI     48
#define DJB__UTIA_NTV      6
#define DJB__UTIA_NPV     48
#define DJB__UTIA_PLANES   3

//---------------------------------------------------------------------------
// Read UTIA BRDF data
utia::utia(const char *filename)
{
	// open file
	std::fstream f(filename, std::fstream::in | std::fstream::binary);
	if (!f.is_open())
		throw exc("djb_error: Failed to open %s\n", filename);

	// allocate memory
	int cnt = DJB__UTIA_PLANES * DJB__UTIA_NTI * DJB__UTIA_NPI
	        * DJB__UTIA_NTV * DJB__UTIA_NPV;
	m_samples.resize(cnt);

	// read data
	f.read((char *)&m_samples[0], sizeof(double) * cnt);

	// normalize
	m_norm = 1.0;
	normalize();
	if (f.fail())
		throw exc("djb_error: Reading %s failed\n", filename);
}

//---------------------------------------------------------------------------
// Look up the UTIA BRDF
vec3 utia::eval(const dir& in, const dir& out) const
{
	double r2d = 180.0 / M_PI;
	double theta_i = r2d * in.theta;
	double theta_o = r2d * out.theta;
	double phi_i = r2d * in.phi;
	double phi_o = r2d * out.phi;

	// make sure we're above horizon
	if (theta_i >= 90.0 || theta_o >= 90.0)
		return djb::vec3(0);

	// make sure phi is in [0, 360)
	while (phi_i < 0.0) {phi_i+= 360.0;}
	while (phi_o < 0.0) {phi_o+= 360.0;}
	while (phi_i >= 360) {phi_i-= 360.0;}
	while (phi_o >= 360) {phi_o-= 360.0;}

	int iti[2], itv[2], ipi[2], ipv[2];
	iti[0] = (int)floor(theta_i / DJB__UTIA_STEP_T);
	iti[1] = iti[0] + 1;
	if(iti[0] > DJB__UTIA_NTI - 2) {
		iti[0] = DJB__UTIA_NTI - 2;
		iti[1] = DJB__UTIA_NTI - 1;
	}
	itv[0] = (int)floor(theta_o / DJB__UTIA_STEP_T);
	itv[1] = itv[0] + 1;
	if(itv[0] > DJB__UTIA_NTV - 2) {
		itv[0] = DJB__UTIA_NTV - 2;
		itv[1] = DJB__UTIA_NTV - 1;
	}

	ipi[0] = (int)floor(phi_i / DJB__UTIA_STEP_P);
	ipi[1] = ipi[0] + 1;
	ipv[0] = (int)floor(phi_o / DJB__UTIA_STEP_P);
	ipv[1] = ipv[0] + 1;

	double sum;
	double wti[2], wtv[2], wpi[2], wpv[2];
	wti[1] = theta_i - (double)(DJB__UTIA_STEP_T * iti[0]);
	wti[0] = (double)(DJB__UTIA_STEP_T * iti[1]) - theta_i;
	sum = wti[0] + wti[1];
	wti[0]/= sum;
	wti[1]/= sum;
	wtv[1] = theta_o - (double)(DJB__UTIA_STEP_T * itv[0]);
	wtv[0] = (double)(DJB__UTIA_STEP_T * itv[1]) - theta_o;
	sum = wtv[0] + wtv[1];
	wtv[0]/= sum;
	wtv[1]/= sum;

	wpi[1] = phi_i - (double)(DJB__UTIA_STEP_P * ipi[0]);
	wpi[0] = (double)(DJB__UTIA_STEP_P * ipi[1]) - phi_i;
	sum = wpi[0] + wpi[1];
	wpi[0]/= sum;
	wpi[1]/= sum;
	wpv[1] = phi_o - (double)(DJB__UTIA_STEP_P * ipv[0]);
	wpv[0] = (double)(DJB__UTIA_STEP_P * ipv[1]) - phi_o;
	sum = wpv[0] + wpv[1];
	wpv[0]/= sum;
	wpv[1]/= sum;

	if(ipi[1] == DJB__UTIA_NPI)
		ipi[1] = 0;
	if(ipv[1] == DJB__UTIA_NPV)
		ipv[1] = 0;

	int nc = DJB__UTIA_NPV * DJB__UTIA_NTV;
	int nr = DJB__UTIA_NPI * DJB__UTIA_NTI;
	double RGB[DJB__UTIA_PLANES];
	for(int isp = 0; isp < DJB__UTIA_PLANES; ++isp) {
		int i, j, k, l;

		RGB[isp] = 0.0;
		for(i = 0; i < 2; ++i)
		for(j = 0; j < 2; ++j)
		for(k = 0; k < 2; ++k)
		for(l = 0; l < 2; ++l) {
			double w = wti[i] * wtv[j] * wpi[k] * wpv[l];
			int idx = isp * nr * nc + nc * (DJB__UTIA_NPI * iti[i] + ipi[k]) 
			        + DJB__UTIA_NPV * itv[j] + ipv[l];

			RGB[isp]+= w * m_samples[idx];
		}
	}
	return vec3(max(0.0, RGB[0]), max(0.0, RGB[1]), max(0.0, RGB[2]));
}

//---------------------------------------------------------------------------
// The UTIA BRDF generates more energy than it receives, and contains 
// negative values for some materials. Therefore, we clamp and 
// scale the data.
void utia::normalize()
{
	int xres = 90;
	int yres = 360;
	double dtheta = M_PI * 0.5 / (double)xres;
	double dphi = 2.0 * M_PI / (double)yres;
	djb::vec3 nint(0);

	// clamp to zero
	for (int i = 0; i < (int)m_samples.size(); ++i) {
#ifndef NVERBOSE
		if (m_samples[i] < 0.0)
			DJB_LOG("djb_verbose: negative UTIA BRDF value found, set to 0\n");
#endif
		m_samples[i] = max(0.0, m_samples[i]);
	}

	// scale
	for (int i = 0; i < xres; ++i)
	for (int j = 0; j < yres; ++j) {
		double tmp1 = (double)i / (double)xres;
		double tmp2 = (double)j / (double)yres;
		double theta = tmp1 * M_PI * 0.5;
		double phi = tmp2 * 2.0 * M_PI;
		dir o(0, 0);
		dir i(theta, phi);
		nint+= eval(i, o) * cos(theta) * sin(theta);
	}
	nint*= dtheta * dphi;

	double mag = max3(nint.x, nint.y, nint.z) * /* magic cst */7.0;
	if (mag > 1.0) {
		double k = 1.0 / mag;
		for (int i = 0; i < (int)m_samples.size(); ++i)
			m_samples[i]*= k;

#ifndef NVERBOSE
		DJB_LOG("djb_verbose: UTIA magnitude = %.9f\n", mag);
#endif
	}
}

// *************************************************************************************************
// Private Spline API
template <typename T>
static T spline_eval(const std::vector<T>& points, double x)
{
	x = min(1.0, max(0.0, x)) * (points.size() - 1); // clamp to edge
	double s1 = floor(x);
	double s2 = min(double(points.size() - 1), ceil(x));
	const T& t1 = points[(int)s1];
	const T& t2 = points[(int)s2];
	double a = x - s1;

	return (1.0 - a) * t1 + a * t2;
}

template <typename T> 
static T
spline2d_eval(
	const std::vector<T>& points,
	int w,
	int h,
	double x,
	double y
) {
	double yabs = fabs(y);
	x = min(1.0, max(0.0, x)) * (w - 1); // clamp to edge
	y = (yabs - floor(yabs)) * (h - 1);  // repeat

	// texcoords
	double x1 = floor(x);
	double x2 = min(x1 + 1.0, (double)(w-1));
	double y1 = floor(y);
	double y2 = min(y1 + 1.0, (double)(h-1));

	// lerp coeffs
	double a1 = (double)x - x1;
	double a2 = (double)y - y1;
	DJB_ASSERT(a1 >= 0.0 && a1 <= 1.0);
	DJB_ASSERT(a2 >= 0.0 && a2 <= 1.0);

	// fetches
	const T& p1 = points[(int)x1 + w * (int)y1];
	const T& p2 = points[(int)x2 + w * (int)y1];
	const T& p3 = points[(int)x1 + w * (int)y2];
	const T& p4 = points[(int)x2 + w * (int)y2];

	// return bilinear interpolation
	double tmp1 = (1.0 - a1) * p1 + a1 * p2;
	double tmp2 = (1.0 - a1) * p3 + a1 * p4;
	return ((1.0 - a2) * tmp1 + a2 * tmp2);
}

// *************************************************************************************************
// Fresnel API implementation
namespace fresnel {

static double unpolarized__eval(double cos_theta, double ior)
{
	double c = cos_theta;
	double n = ior;
	double g = sqrt(n * n + c * c - 1.0);
	double tmp1 = c * (g + c) - 1.0;
	double tmp2 = c * (g - c) + 1.0;
	double tmp3 = (tmp1 * tmp1) / (tmp2 * tmp2);
	double tmp4 = ((g - c) * (g - c)) / ((g + c) * (g + c));

	return ((0.5 * tmp4) * (1.0 + tmp3));
}

vec3 unpolarized::eval(double theta) const
{
	DJB_ASSERT(theta >= 0 && theta <= M_PI * 0.5 && "Invalid Angle");
	double cos_theta = cos(theta);
	double F[3];

	for (int i = 0; i < 3; ++i)
		F[i] = unpolarized__eval(cos_theta, vec3::to_raw(ior)[i]);

	return vec3::from_raw(F);
}

vec3 schlick::eval(double theta) const
{
	DJB_ASSERT(theta >= 0 && theta <= M_PI * 0.5 && "Invalid Angle");
	double c1 = 1.0 - cos(theta);
	double c2 = c1 * c1;
	double c5 = c2 * c2 * c1;

	return f0 + c5 * (vec3(1) - f0);
};

vec3 sgd::eval(double theta) const
{
	DJB_ASSERT(theta >= 0 && theta <= M_PI * 0.5 && "Invalid Angle");
	double c = cos(theta);

	return f0 - c * f1 + pow(1.0 - c, 5.0) * (vec3(1) - f0);
};

vec3 spline::eval(double theta) const
{
	DJB_ASSERT(theta >= 0 && theta <= M_PI * 0.5 && "Invalid Angle");
	return spline_eval(m_points, theta * 2.0 / M_PI);
}

} // namespace fresnel

// *************************************************************************************************
// Microfacet API implementation

// -------------------------------------------------------------------------------------------------
/**
 * Compute the covariance matrix associated to the parameters of an ellipse
 *
 * The matrix is {{m11, m12}, {m12, m22}}
 */
static void
ellipse_to_covariance(
	double r1, double r2, double angle,
	double *m11, double *m22, double *m12
) {
	double cosAngle = cos(angle);
	double sinAngle = sin(angle);
	double r1Sqr = r1 * r1;
	double r2Sqr = r2 * r2;
	double M[3]; // inverse cov matrix
	double mdet; // inverse cov matrix determinant

	// compute the inverse of the covariance matrix from 
	// the parameters of the ellipse
	M[0] = cosAngle*cosAngle/r1Sqr + sinAngle*sinAngle/r2Sqr;
	M[1] = sinAngle*sinAngle/r1Sqr + cosAngle*cosAngle/r2Sqr;
	M[2] = cosAngle*sinAngle*(1.0/r1Sqr - 1.0/r2Sqr); // inverse covariance term
	mdet = M[0] * M[1] - M[2] * M[2]; // determinant for inversion

	// produce output from the terms of the inverse of M
	(*m11) = M[1] / mdet;  // Var(X)
	(*m22) = M[0] / mdet;  // Var(Y)
	(*m12) =-M[2] / mdet;  // Cov(X,Y)
}


// -------------------------------------------------------------------------------------------------
/**
 * Compute the roughness parameters associated a covariance matrix
 *
 * The matrix is {{m11, m12}, {m12, m22}}
 */
static void
covariance_to_ellipse(
	double m11, double m22, double m12,
	double *r1, double *r2, double *angle
) {
	if (m12 == 0.0) {
		(*r1) = sqrt(m11);
		(*r2) = sqrt(m22);
		(*angle) = 0.0;
	} else {
		double tmp1 = m11 - m22;
		double tmp2 = sqrt(tmp1 * tmp1 + 4.0 * m12 * m12);
		double tmp3 = (m11 - m22 - tmp2) / (m12 * 2.0);
		(*r1) = sqrt(0.5 * (m11 + m22 - tmp2));
		(*r2) = sqrt(0.5 * (m11 + m22 + tmp2));
		(*angle) = atan2(1.0, tmp3);
	}
}


//---------------------------------------------------------------------------
// Facet Ctor
microfacet::microfacet(
	int gaf,
	double r1, double r2, double rangle, 
	const fresnel::impl& f,
	bool shadow
):
	m_fresnel(f.copy()),
	m_r1(r1), m_r2(r2), m_rangle(rangle),
	m_m11(1), m_m22(1), m_m12(0),
	m_sigma1(1), m_sigma2(1),
	m_rho(0), m_sqrt_one_minus_rho_sqr(0),
	m_gaf(gaf), m_shadow(shadow)
{
	set_gaf(gaf);
	set_roughness(r1, r2, rangle);
}

microfacet::microfacet(int gaf, double r, const fresnel::impl& f, bool shadow):
	m_fresnel(f.copy()),
	m_r1(r), m_r2(r), m_rangle(0),
	m_m11(1), m_m22(1), m_m12(0),
	m_sigma1(1), m_sigma2(1),
	m_rho(0), m_sqrt_one_minus_rho_sqr(0),
	m_gaf(gaf), m_shadow(shadow)
{
	set_gaf(gaf);
	set_roughness(r);
}

//---------------------------------------------------------------------------
// Accessors
void microfacet::get_covariance(double *var1, double *var2, double *cov) const
{
	if (var1) (*var1) = m_m11;
	if (var2) (*var2) = m_m22;
	if (cov)  (*cov)  = m_m12;
}

void microfacet::get_roughness(double *r1, double *r2, double *rangle) const
{
	if (r1) (*r1) = m_r1;
	if (r2) (*r2) = m_r2;
	if (rangle) (*rangle) = m_rangle;
}

//---------------------------------------------------------------------------
// Mutators
void microfacet::set_gaf(int gaf)
{
	if (gaf == GAF_NONE || gaf == GAF_VGROOVE || gaf == GAF_SMITH) {
		m_gaf = gaf;
	} else {
		throw exc("djb_error: Invalid GAF\n");
	}
}

void microfacet::set_fresnel(const fresnel::impl& f)
{
	delete m_fresnel;
	m_fresnel = f.copy();
}

void microfacet::set_roughness(double r1, double r2, double rangle)
{
	DJB_ASSERT(r1 > 0 && r2 > 0);
	m_r1 = r1;
	m_r2 = r2;
	m_rangle = rangle;
	ellipse_to_covariance(m_r1, m_r2, m_rangle, &m_m11, &m_m22, &m_m12);
	m_det = m_m11 * m_m22 - m_m12 * m_m12;
	m_sigma1 = sqrt(m_m11);
	m_sigma2 = sqrt(m_m22);
	m_rho = m_m12 / (m_sigma1 * m_sigma2);
	DJB_ASSERT(m_rho < 1.0 && m_rho > -1.0);
	m_sqrt_one_minus_rho_sqr = sqrt(1.0 - m_rho * m_rho);
}

//---------------------------------------------------------------------------
// Normal Mapped Microfacet GAF
double microfacet::gaf_nmap(const dir& i, const dir& o, const dir& h) const
{
	double cos_theta_h = cos(h.theta);
	double cos_theta_o = cos(o.theta);
	double cos_theta_d = cos_theta_h * cos_theta_o
	                   + cos(h.phi - o.phi) * sin(h.theta) * sin(o.theta);
	double masking = cos_theta_o * fabs(cos_theta_h / cos_theta_d);

	if (m_shadow) {
		double shadowing = i.theta < M_PI * 0.5 ? 1.0 : 0.0;

		return masking * shadowing;
	}

	return masking;
}

//---------------------------------------------------------------------------
// V Grooved Microfacet GAF
double microfacet::gaf_vgroove(const dir& i, const dir& o, const dir& h) const
{
	if (h.theta < 0.5 * M_PI && o.theta < 0.5 * M_PI) {
		double cos_theta_h = cos(h.theta);
		double sin_theta_h = sin(h.theta);
		double cos_theta_o = cos(o.theta);
		double cos_theta_d = cos_theta_h * cos_theta_o
		                   + cos(h.phi - o.phi) * sin_theta_h 
		                   * sin(o.theta);
		double tmp = 2.0 * cos_theta_o * cos_theta_h / cos_theta_d;
		double masking = min(1.0, tmp);

		if (m_shadow) {
			if (i.theta < 0.5 * M_PI) {
				double cos_theta_i = cos(i.theta);
				double cos_theta_d = cos_theta_h * cos_theta_i
				                   + cos(h.phi - i.phi) * sin_theta_h
				                   * sin(i.theta);

				double tmp = 2.0 * cos_theta_i * cos_theta_h / cos_theta_d;
				double masking_and_shadowing = min(masking, tmp);

				return masking_and_shadowing;
			}

			return 0.0; // below horizon
		}

		return masking;
	}

	return 0.0; // below horizon
}

//---------------------------------------------------------------------------
// Smith Height Correlated Microfacet GAF
double microfacet::gaf_smith(const dir& i, const dir& o, const dir& h) const
{
	double g1_o = g1(o);

	if (m_shadow) {
		double g1_i = g1(i);
		double tmp = g1_i * g1_o;

		if (tmp > 0.0)
			return (tmp / (g1_i + g1_o - tmp));

		return 0.0; // fully shadowed
	}

	return g1_o;
}

//---------------------------------------------------------------------------
// Microfacet GAF
double microfacet::gaf(const dir& i, const dir& o, const dir& h) const
{
	switch (m_gaf) {
		case GAF_NONE: return gaf_nmap(i, o, h);
		case GAF_VGROOVE: return gaf_vgroove(i, o, h);
		case GAF_SMITH: return gaf_smith(i, o, h);
		default: abort(); // should never happen
	}
}

//---------------------------------------------------------------------------
// Microfacet NDF
double microfacet::ndf(const dir& h) const
{
	return (p22(h) / pow(cos(h.theta), 4.0));
}

//---------------------------------------------------------------------------
// Evaluate Projected Microfacet BRDF
vec3 microfacet::evalp(const dir& i, const dir& o) const
{
	dir h, d;
	std_coords_to_half_diff_coords(i, o, h, d);

	vec3 F = eval_fresnel(d.theta);
	double D = ndf(h);
	double G = gaf(i, o, h);

	return (F * D * G) / (4.0 * cos(o.theta));
}

//---------------------------------------------------------------------------
// Evaluate Microfacet BRDF
vec3 microfacet::eval(const dir& i, const dir& o) const
{
	vec3 fr_p = evalp(i, o);
	return fr_p / cos(i.theta);
}

//---------------------------------------------------------------------------
// Set covariance terms
void microfacet::set_covariance(double m11, double m22, double m12)
{
	DJB_ASSERT(m11 > 0 && m22 > 0 && "Invalid Variance\n");
	DJB_ASSERT(m11 * m22 - m12 * m12 > 0 && "Invalid Cov Determinant\n");
	double r1, r2, rangle;
	covariance_to_ellipse(m11, m22, m12, &r1, &r2, &rangle);
	set_roughness(r1, r2, rangle);
}

// *************************************************************************************************
// Radial Microfacet API implementation

dir radial::sample_nmap(const dir& out, double u1, double u2) const
{
	double mag = qf_radial(u1);
	double phi = u2 * 2.0 * M_PI;
	double std1 = mag * cos(phi);
	double std2 = mag * sin(phi);
	double sx = std1 * m_sigma1;
	double sy = (std1 * m_rho + std2 * m_sqrt_one_minus_rho_sqr) * m_sigma2;

	return dir(normalize(vec3(-sx, -sy, 1)));
}

dir radial::sample_vgroove(const dir& out, double u1, double u2) const
{
	double mag = qf_radial(u1);
	double phi = u2 * 2.0 * M_PI;
	double std1 = mag * cos(phi);
	double std2 = mag * sin(phi);
	double sx = std1 * m_sigma1;
	double sy = (std1 * m_rho + std2 * m_sqrt_one_minus_rho_sqr) * m_sigma2;
	vec3 m1 = normalize(vec3(-sx, -sy, 1));
	vec3 m2 = normalize(vec3(+sx, +sy, 1));
	vec3 o = vec3(out);
	double tmp1 = max(0.0, dot(o, m1));
	double tmp2 = max(0.0, dot(o, m2));
	double tmp3 = tmp2 / (tmp1 + tmp2);

	if (u1 > tmp3) // another random number should be used here...
		return dir(m1);
	else
		return dir(m2);
}

dir radial::sample_smith(const dir& out, double u1, double u2) const
{
	return sample_nmap(out, u1, u2);
}

dir radial::sample(const dir& out, double u1, double u2) const
{
	switch (m_gaf) {
		case GAF_NONE:    return sample_nmap(out, u1, u2);
		case GAF_VGROOVE: return sample_vgroove(out, u1, u2);
		case GAF_SMITH:   return sample_smith(out, u1, u2);
		default: abort(); // should never happen
	}
}

double radial::pdf_nmap(const dir& out, const dir& m) const
{
	return (p22(m) / pow(cos(m.theta), 3.0));
}

double radial::pdf_vgroove(const dir& out, const dir& m) const
{
	double G1 = gaf_vgroove(out, out, m);
	double D = ndf(m);
	vec3 vo = normalize(vec3(out));
	vec3 vm = normalize(vec3(m));

	return (G1 * D * max(0.0, dot(vo, vm)) / cos(out.theta));
}

double radial::pdf_smith(const dir& out, const dir& m) const
{
	return pdf_nmap(out, m);
}

double radial::pdf(const dir& out, const dir& m) const
{
	switch (m_gaf) {
		case GAF_NONE:    return pdf_nmap(out, m);
		case GAF_VGROOVE: return pdf_vgroove(out, m);
		case GAF_SMITH:   return pdf_smith(out, m);
		default: abort(); // should never happen
	}
}

double radial::p22(const dir& h) const
{
	double tan_theta = tan(h.theta);
	double x1 = -tan_theta * cos(h.phi);
	double x2 = -tan_theta * sin(h.phi);
	double z = (x1*x1*m_m22 + x2*x2*m_m11 - 2.0*x1*x2*m_m12) / m_det;

	return p22_radial(z) / sqrt(m_det);
}

double radial::g1(const dir& k) const
{
	if (k.theta < M_PI * 0.5) {
		if (k.theta > 0.0) {
			double cos_phi = cos(k.phi);
			double sin_phi = sin(k.phi);
			double var = m_m11 * cos_phi * cos_phi
			           + m_m22 * sin_phi * sin_phi
			           + m_m12 * 2.0 * cos_phi * sin_phi;
			double z = tan(M_PI * 0.5 - k.theta) / sqrt(2.0 * var);

			return g1_radial(z);
		}
		return 1.0; // normal incidence: no shadowing
	}
	return 0.0; // horizon (or below)
}

double radial::p22_explicit(const dir& h, double r) const
{
	return p22_explicit(h, r, r);
}

double
radial::p22_explicit(const dir& h, double r1, double r2, double rangle) const
{
	DJB_ASSERT(r1 > 0.0 && r2 > 0.0);
	double m11, m22, m12;
	double tan_theta = tan(h.theta);
	double x1 = -tan_theta * cos(h.phi);
	double x2 = -tan_theta * sin(h.phi);
	ellipse_to_covariance(r1, r2, rangle, &m11, &m22, &m12);
	double det = m11 * m22 - m12 * m12;
	double z = (x1*x1*m22 + x2*x2*m11 - 2.0*x1*x2*m12) / det;
	double p22 = p22_radial(z) / sqrt(det);

	return p22;
}

double radial::ndf_explicit(const dir& h, double r) const
{
	return ndf_explicit(h, r, r);
}

double
radial::ndf_explicit(const dir& h, double r1, double r2, double rangle) const
{
	double p22 = p22_explicit(h, r1, r2, rangle);

	return (p22 / pow(cos(h.theta), 4.0));
}

dir
radial::sample_explicit(const dir& out, double u1, double u2, double r) const
{
	return sample_explicit(out, u1, u2, r, r);
}

dir
radial::sample_explicit(
	const dir& out, 
	double u1, double u2, 
	double r1, double r2, double rangle
) const {
	double m11, m22, m12;
	double mag = qf_radial(u1);
	double phi = u2 * 2.0 * M_PI;
	double std1 = mag * cos(phi);
	double std2 = mag * sin(phi);
	ellipse_to_covariance(r1, r2, rangle, &m11, &m22, &m12);
	double sigma1 = sqrt(m11);
	double sigma2 = sqrt(m22);
	double rho = m12 / (sigma1 * sigma2);
	double sx = std1 * sigma1;
	double sy = (std1 * rho + std2 * sqrt(1.0 - rho * rho)) * sigma2;

	return dir(normalize(vec3(-sx, -sy, 1)));
}

vec3 radial::evalp_explicit(const dir& i, const dir& o, double r) const
{
	return evalp_explicit(i, o, r, r);
}

vec3
radial::evalp_explicit(
	const dir& i, const dir& o, 
	double r1, double r2, double rangle
) const {
	dir h, d;
	std_coords_to_half_diff_coords(i, o, h, d);

	vec3 F = eval_fresnel(d.theta);
	double D = ndf_explicit(h, r1, r2, rangle);
	double G = gaf_explicit(i, o, h, r1, r2, rangle);

	return (F * D * G) / (4.0 * cos(o.theta));
}

double
radial::gaf_explicit(const dir& i, const dir& o, const dir& h, double r) const
{
	return gaf_explicit(i, o, h, r, r);
}

double
radial::gaf_explicit(
	const dir& i, const dir& o, const dir& h,
	double r1, double r2, double rangle
) const {
	double g1_o = g1_explicit(o, r1, r2, rangle);

	if (m_shadow) {
		double g1_i = g1_explicit(i, r1, r2, rangle);
		double tmp = g1_i * g1_o;

		if (tmp > 0.0)
			return (tmp / (g1_i + g1_o - tmp));

		return 0.0; // fully shadowed
	}

	return g1_o;
}

double radial::g1_explicit(const dir& k, double r) const
{
	return g1_explicit(k, r, r);
}

double
radial::g1_explicit(const dir& k, double r1, double r2, double rangle) const
{
	if (k.theta < M_PI * 0.5) {
		if (k.theta > 0.0) {
			double m11, m22, m12;
			ellipse_to_covariance(r1, r2, rangle, &m11, &m22, &m12);
			double cos_phi = cos(k.phi);
			double sin_phi = sin(k.phi);
			double var = m11 * cos_phi * cos_phi
			           + m22 * sin_phi * sin_phi
			           + m12 * 2.0 * cos_phi * sin_phi;
			double z = tan(M_PI * 0.5 - k.theta) / sqrt(2.0 * var);

			return g1_radial(z);
		}
		return 1.0; // normal incidence: no shadowing
	}
	return 0.0; // horizon (or below)
}

// *************************************************************************************************
// Gaussian Microfacet API implementation

double gaussian::p22_radial(double z) const
{
	return (exp(-0.5 * z) / (2.0 * M_PI));
}

double gaussian::g1_radial(double z) const
{
	z/= sqrt(2.0);
	double lambda = 0.5 * (exp(-z*z) / (z * sqrt(M_PI)) - 1.0 + erf(z));
	return (1.0 / (1.0 + lambda));
}

double gaussian::cdf_radial(double theta) const
{
	double z = tan(theta);
	return (1.0 - exp(-0.5 * z * z));
}

double gaussian::qf_radial(double r) const
{
	return sqrt(-2.0 * log(1.0 - r));
}

// *************************************************************************************************
// GGX Microfacet API implementation

double ggx::p22_radial(double z) const
{
	const double inv_pi = 1.0 / M_PI;
	double den = 1.0 + z;

	return (inv_pi / (den * den));
}

double ggx::g1_radial(double z) const
{
	double lambda = 0.5 * (sqrt(1.0 + 1.0 / (z * z)) - 1.0);
	return (1.0 / (1.0 + lambda));
}

double ggx::cdf_radial(double theta) const
{
	double z = tan(theta);
	double tmp = z * z;
	return (tmp / (1.0 + tmp));
}

double ggx::qf_radial(double r) const
{
	return sqrt(r / (1.0 - r));
}

// *************************************************************************************************
// Tabulated Microfacet API implementation

double tabular::p22_radial(double z) const
{
	double x = sqrt(2.0 * atan(sqrt(z)) / M_PI);
	return spline_eval(m_p22, x);
}

double tabular::g1_radial(double z) const
{
	double x = 1.0 - 2.0 * atan(z) / M_PI;
	return spline_eval(m_g1, x);
}

double tabular::cdf_radial(double theta) const
{
	double z = theta * 2.0 / M_PI;
	return spline_eval(m_cdf, sqrt(z));
}

double tabular::qf_radial(double r) const
{
	double z = spline_eval(m_qf, r);
	return tan(z * 0.5 * M_PI);
}

double tabular_anisotropic::p22(const dir& k) const
{
	double slope = tan(k.theta);
	double sx = slope * cos(k.phi - m_rangle) / m_r1;
	double sy = slope * sin(k.phi - m_rangle) / m_r2;
	dir k2 = dir(normalize(vec3(-sx, -sy, 1)));
	if (k2.phi < 0.0) k2.phi+= 2.0 * M_PI;
	double x = k2.theta * 2.0 / M_PI;
	double y = k2.phi * 0.5 / M_PI;
	int w = m_elevation_res;
	int h = m_azimuthal_res;

	return spline2d_eval<double>(m_p22, w, h, x, y) / sqrt(m_det);
}

double tabular_anisotropic::g1(const dir& k) const
{
	double slope = tan(k.theta);
	double sx = slope * cos(k.phi - m_rangle) * m_r1;
	double sy = slope * sin(k.phi - m_rangle) * m_r2;
	dir k2 = dir(normalize(vec3(-sx, -sy, 1)));
	if (k2.phi < 0.0) k2.phi+= 2.0 * M_PI;
	double x = k2.theta * 2.0 / M_PI;
	double y = k2.phi * 0.5 / M_PI;
	int w = m_elevation_res;
	int h = m_azimuthal_res;

	return spline2d_eval<double>(m_g1, w, h, x, y);
}

double tabular_anisotropic::p22_explicit(const dir& h, double r) const
{
	return ndf_explicit(h, r, r);
}

double
tabular_anisotropic::p22_explicit(
	const dir& k,
	double r1,
	double r2,
	double rangle
) const {
	DJB_ASSERT(r1 > 0.0 && r2 > 0.0);
	double m11, m22, m12;
	ellipse_to_covariance(r1, r2, rangle, &m11, &m22, &m12);
	double det = m11 * m22 - m12 * m12;
	double slope = tan(k.theta);
	double sx = slope * cos(k.phi - rangle) / r1;
	double sy = slope * sin(k.phi - rangle) / r2;
	dir k2 = dir(normalize(vec3(-sx, -sy, 1)));
	if (k2.phi < 0.0) k2.phi+= 2.0 * M_PI;
	double x = k2.theta * 2.0 / M_PI;
	double y = k2.phi * 0.5 / M_PI;
	int w = m_elevation_res;
	int h = m_azimuthal_res;

	return spline2d_eval<double>(m_p22, w, h, x, y) / sqrt(det);
}

double tabular_anisotropic::ndf_explicit(const dir& h, double r) const
{
	return ndf_explicit(h, r, r);
}

double
tabular_anisotropic::ndf_explicit(
	const dir& h,
	double r1,
	double r2,
	double rangle
) const {
	double p22 = p22_explicit(h, r1, r2, rangle);

	return (p22 / pow(cos(h.theta), 4.0));
}

vec3 
tabular_anisotropic::evalp_explicit(const dir& i, const dir& o, double r) const
{
	return evalp_explicit(i, o, r, r);
}

vec3
tabular_anisotropic::evalp_explicit(
	const dir& i, const dir& o,
	double r1, double r2, double rangle
) const {
	dir h, d;
	std_coords_to_half_diff_coords(i, o, h, d);

	vec3 F = eval_fresnel(d.theta);
	double D = ndf_explicit(h, r1, r2, rangle);
	double G = gaf_explicit(i, o, h, r1, r2, rangle);

	return (F * D * G) / (4.0 * cos(o.theta));
}

double
tabular_anisotropic::gaf_explicit(
	const dir& i,
	const dir& o,
	const dir& h,
	double r
) const {
	return gaf_explicit(i, o, h, r, r);
}

double
tabular_anisotropic::gaf_explicit(
	const dir& i, const dir& o, const dir& h,
	double r1, double r2, double rangle
) const {
	double g1_o = g1_explicit(o, r1, r2, rangle);

	if (m_shadow) {
		double g1_i = g1_explicit(i, r1, r2, rangle);
		double tmp = g1_i * g1_o;

		if (tmp > 0.0)
			return (tmp / (g1_i + g1_o - tmp));

		return 0.0; // fully shadowed
	}

	return g1_o;
}

double tabular_anisotropic::g1_explicit(const dir& k, double r) const
{
	return g1_explicit(k, r, r);
}

double
tabular_anisotropic::g1_explicit(
	const dir& k,
	double r1,
	double r2,
	double rangle
) const {
	double slope = tan(k.theta);
	double sx = slope * cos(k.phi - rangle) * r1;
	double sy = slope * sin(k.phi - rangle) * r2;
	dir k2 = dir(normalize(vec3(-sx, -sy, 1)));
	if (k2.phi < 0.0) k2.phi+= 2.0 * M_PI;
	double x = k2.theta * 2.0 / M_PI;
	double y = k2.phi * 0.5 / M_PI;
	int w = m_elevation_res;
	int h = m_azimuthal_res;

	return spline2d_eval<double>(m_g1, w, h, x, y);
}
#if 0 //TODO
dir
tabular_anisotropic::sample_explicit(
	const dir& out,
	double u1, double u2,
	double r
) const {
	return sample_explicit(out, u1, u2, r, r);
}

dir
tabular_anisotropic::sample_explicit(
	const dir& out, 
	double u1, double u2, 
	double r1, double r2, double rangle
) const {
	double m11, m22, m12;
	double mag = qf_radial(u1);
	double phi = u2 * 2.0 * M_PI;
	double std1 = mag * cos(phi);
	double std2 = mag * sin(phi);
	ellipse_to_covariance(r1, r2, rangle, &m11, &m22, &m12);
	double sigma1 = sqrt(m11);
	double sigma2 = sqrt(m22);
	double rho = m12 / (sigma1 * sigma2);
	double sx = std1 * sigma1;
	double sy = (std1 * rho + std2 * sqrt(1.0 - rho * rho)) * sigma2;

	return dir(normalize(vec3(-sx, -sy, 1)));
}
#endif
// -------------------------------------------------------------------------------------------------
// Constructors
tabular::tabular(int gaf, const brdf& brdf, int res, bool shadow):
	radial(gaf, 1)
{
	DJB_ASSERT(res > 2 && "Invalid Resolution");
	m_shadow = shadow;

	m_p22.reserve(res);
	m_g1.reserve(res);
	m_cdf.reserve(res);
	m_qf.reserve(res);

	switch (gaf) {
		case microfacet::GAF_NONE: compute_p22_nmap(brdf, res); break;
		case microfacet::GAF_VGROOVE: compute_p22_vgroove(brdf, res); break;
		case microfacet::GAF_SMITH: compute_p22_smith(brdf, res); break;
		default: abort();
	}
	normalize_p22();
	compute_g1();

	compute_fresnel(brdf, res);
	compute_cdf();
	compute_qf();
}

tabular_anisotropic::tabular_anisotropic(
	int gaf, const brdf& brdf,
	int elevation_res,
	int azimuthal_res,
	bool shadow
): microfacet(gaf, 1)
{
	DJB_ASSERT(elevation_res > 2 && azimuthal_res > 2 && "Invalid Resolution");
	m_shadow = shadow;

	m_elevation_res = elevation_res;
	m_azimuthal_res = azimuthal_res;

	m_p22.reserve(elevation_res * azimuthal_res);
	m_g1.reserve(elevation_res * azimuthal_res);
	m_pdf1.reserve(azimuthal_res);
	m_cdf1.reserve(azimuthal_res);
	m_qf1.reserve(azimuthal_res);
	m_pdf2.reserve(elevation_res * azimuthal_res);
	m_cdf2.reserve(elevation_res * azimuthal_res);
	m_qf2.reserve(elevation_res * azimuthal_res);

	switch (gaf) {
		case microfacet::GAF_NONE: compute_p22_nmap(brdf); break;
		case microfacet::GAF_VGROOVE: compute_p22_vgroove(brdf); break;
		case microfacet::GAF_SMITH: compute_p22_smith(brdf); break;
		default: abort();
	}
	normalize_p22();
	compute_g1();
	compute_fresnel(brdf, m_elevation_res);
	compute_pdf1();
	compute_cdf1();
	compute_qf1();
	compute_pdf2();
	compute_cdf2();
	compute_qf2();
}

// -------------------------------------------------------------------------------------------------
// Normalize the slope pdf
void tabular::normalize_p22()
{
	const double dphi = 2.0 * M_PI;
	const int ntheta = 128;
	double dtheta = sqrt(M_PI * 0.5) / (double)ntheta;
	double k = 0.0;

	for (int i = 0; i < ntheta; ++i) {
		double tmp = (double)i / (double)ntheta; // in [0,1)
		double theta = tmp * sqrt(M_PI * 0.5);
		double theta_sqr = theta * theta; // in [0, pi/2)
		double c = cos(theta_sqr);
		double pdf = p22(dir(theta_sqr, 0.0));

		k+= theta * (pdf * tan(theta_sqr)) / (c * c);
	}
	k*= (2.0 * dtheta) * dphi;

	// normalize the slope pdf
	k = 1.0 / k;
	for (int i = 0; i < (int)m_p22.size(); ++i)
		m_p22[i]*= k;

#ifndef NVERBOSE
	DJB_LOG("djb_verbose: Slope PDF norm. constant = %.9f\n", k);
#endif
}

void tabular_anisotropic::normalize_p22()
{
	const int ntheta = 128;
	const int nphi   = 256;
	double dtheta = sqrt(0.5 * M_PI) / (double)ntheta;
	double dphi   = 2.0 * M_PI / (double)nphi;
	double k = 0.0;

	for (int j = 0; j < nphi; ++j) {
		double tmp = (double)j / (double)nphi;
		double phi = tmp * 2.0 * M_PI;
		for (int i = 0; i < ntheta; ++i) {
			double tmp = (double)i / (double)ntheta; // in [0,1)
			double theta = tmp * sqrt(M_PI * 0.5);
			double theta_sqr = theta * theta; // in [0, pi/2)
			double c = cos(theta_sqr);
			double pdf = p22(dir(theta_sqr, phi));
			double weight = (theta * tan(theta_sqr)) / (c * c);

			k+= (weight * pdf);
		}
	}
	k*= (2.0 * dtheta * dphi);

	// normalize the slope pdf
	k = 1.0 / k;
	for (int i = 0; i < (int)m_p22.size(); ++i)
		m_p22[i]*= k;

#ifndef NVERBOSE
	DJB_LOG("djb_verbose: Anisotropic slope PDF norm. constant = %.9f\n", k);
#endif
}

// -------------------------------------------------------------------------------------------------
/**
 * Compute the Smith Masking Term
 *
 * This requires the computation of an integral.
 * The resolution at which the integral is computed is hard coded to 
 * yield sufficient precision for most distributions.
 */
void tabular::compute_g1()
{
	const int ntheta = 90;
	const int nphi   = 180;
	double dtheta = sqrt(M_PI * 0.5) / (double)ntheta;
	double dphi   = 2.0 * M_PI / (double)nphi;
	int cnt = m_p22.size() - 1;

	for (int i = 0; i < cnt; ++i) {
		double tmp = (double)i / (double)cnt; // in [0, 1)
		double theta_o = tmp * 0.5 * M_PI; // in [0, pi/2)
		double cos_theta_o = cos(theta_o);
		double nint = 0.0;

		for (int j2 = 0; j2 < nphi; ++j2) {
			double tmp = (double)j2 / (double)nphi; // in [0, 1)
			double phi = tmp * 2.0 * M_PI;          // in [0, 2pi)
			for (int j1 = 0; j1 < ntheta; ++j1) {
				double tmp = (double)j1 / (double)ntheta; // in [0, 1)
				double theta = tmp * sqrt(M_PI * 0.5); // in [0, sqrt(pi/2))
				double theta_sqr = theta * theta; // in [0, pi/2)
				double sin_theta = sin(theta_sqr);
				double m_dot_o = sin(theta_o) * sin_theta * cos(phi)
					           + cos_theta_o * cos(theta_sqr);

				nint+= ndf(dir(theta_sqr, phi))
					 * max(0.0, m_dot_o)
					 * theta * sin_theta;
			}
		}
		nint*= 2.0 * dtheta * dphi;
		m_g1.push_back(min(1.0, cos_theta_o / nint));
	}
	m_g1.push_back(0);

#ifndef NVERBOSE
	DJB_LOG("djb_verbose: Smith masking term ready\n");
#endif
}

void tabular_anisotropic::compute_g1()
{
	const int ntheta = 45;
	const int nphi   = 90;
	double dtheta = sqrt(M_PI * 0.5) / (double)ntheta;
	double dphi   = 2.0 * M_PI / (double)nphi;
	int w = m_elevation_res - 1;
	int h = m_azimuthal_res;

	for (int i2 = 0; i2 < h; ++i2) {
		double tmp = (double)i2 / (double)h; // in [0, 1)
		double phi_o = tmp * 2.0 * M_PI; // in [0, 2pi)
		for (int i1 = 0; i1 < w; ++i1) {
			double tmp = (double)i1 / (double)w; // in [0, 1)
			double theta_o = tmp * 0.5 * M_PI; // in [0, pi/2)
			double cos_theta_o = cos(theta_o);
			double nint = 0.0;

			for (int j2 = 0; j2 < nphi; ++j2) {
				double tmp = (double)j2 / (double)nphi; // in [0, 1)
				double phi = tmp * 2.0 * M_PI;          // in [0, 2pi)
				for (int j1 = 0; j1 < ntheta; ++j1) {
					double tmp = (double)j1 / (double)ntheta; // in [0, 1)
					double theta = tmp * sqrt(M_PI * 0.5); // in [0, sqrt(pi/2))
					double theta_sqr = theta * theta; // in [0, pi/2)
					double sin_theta = sin(theta_sqr);
					double m_dot_o = sin(theta_o) * sin_theta * cos(phi - phi_o)
					               + cos_theta_o * cos(theta_sqr);
					double weight = theta * sin_theta;
					double masking = max(0.0, m_dot_o)
					               * ndf(dir(theta_sqr, phi));

					nint+= weight * masking;
				}
			}
			nint*= 2.0 * dtheta * dphi;
			m_g1.push_back(min(1.0, cos_theta_o / nint));
		}
		m_g1.push_back(0);
	}

#ifndef NVERBOSE
	DJB_LOG("djb_verbose: Anisotropic Smith masking term ready\n");
#endif
}

// -------------------------------------------------------------------------------------------------
// Inversion Assuming a Normal Mapped Microfacet BRDF
void tabular::compute_p22_nmap(const brdf& brdf, int res)
{
	int cnt = res - 1;

	for (int i = 0; i < cnt; ++i) {
		double tmp = (double)i / (double)cnt; // in [0, 1)
		double theta = tmp * sqrt(M_PI * 0.5);
		double theta_sqr = theta * theta; // in [0, pi/2)
		double cos_theta = cos(theta_sqr);
		double cos_theta_sqr = cos_theta * cos_theta;
		vec3 fr = brdf.eval(dir(theta_sqr, 0), dir(theta_sqr, 0));
		double fr_i = fr.intensity();
		double pdf = (cos_theta_sqr * cos_theta_sqr) * (4.0 * fr_i);

		m_p22.push_back(pdf);
	}
	m_p22.push_back(0);
}

void tabular_anisotropic::compute_p22_nmap(const brdf& brdf)
{
	int w = m_elevation_res - 1;
	int h = m_azimuthal_res;

	for (int j = 0; j < h; ++j) {
		double tmp = (double)j / (double)h;
		double phi = tmp * 2.0 * M_PI;
		for (int i = 0; i < w; ++i) {
			double tmp = (double)i / (double)w; // in [0, 1)
			double theta = tmp * 0.5 * M_PI; // in [0, 2pi)
			double cos_theta = cos(theta);
			double cos_theta_sqr = cos_theta * cos_theta;
			vec3 fr = brdf.eval(dir(theta, phi), dir(theta, phi));
			double fr_i = fr.intensity();
			double pdf = (cos_theta_sqr * cos_theta_sqr) * (4.0 * fr_i);

			m_p22.push_back(pdf);
		}
		m_p22.push_back(0); // for lerping
	}
}

// -------------------------------------------------------------------------------------------------
// Inversion Assuming a V Grooved Cavities Microfacet BRDF
void tabular::compute_p22_vgroove(const brdf& brdf, int res)
{
	int cnt = res - 1;

	for (int i = 0; i < cnt; ++i) {
		double tmp = (double)i / (double)cnt; // in [0, 1)
		double theta = tmp * sqrt(M_PI * 0.5);
		double theta_sqr = theta * theta; // in [0, pi/2)
		double cos_theta = cos(theta_sqr);
		double cos_theta_sqr = cos_theta * cos_theta;
		double gaf = max(0.5, cos_theta_sqr);
		vec3 fr = brdf.eval(dir(theta_sqr, 0), dir(theta_sqr, 0));
		double fr_i = fr.intensity();
		double pdf = gaf * (cos_theta_sqr * cos_theta_sqr) * (4.0 * fr_i);

		m_p22.push_back(pdf);
	}
	m_p22.push_back(0);
}

void tabular_anisotropic::compute_p22_vgroove(const brdf& brdf)
{
	int w = m_elevation_res - 1;
	int h = m_azimuthal_res;

	for (int j = 0; j < h; ++j) {
		double tmp = (double)j / (double)h;
		double phi = tmp * 2.0 * M_PI;
		for (int i = 0; i < w; ++i) {
			double tmp = (double)i / (double)w; // in [0, 1)
			double theta = tmp * 0.5 * M_PI; // in [0, 2pi)
			double cos_theta = cos(theta);
			double cos_theta_sqr = cos_theta * cos_theta;
			double gaf = max(0.5, cos_theta_sqr);
			vec3 fr = brdf.eval(dir(theta, phi), dir(theta, phi));
			double fr_i = fr.intensity();
			double pdf = gaf * (cos_theta_sqr * cos_theta_sqr) * (4.0 * fr_i);

			m_p22.push_back(pdf);
		}
		m_p22.push_back(0); // for lerping
	}
}

// -------------------------------------------------------------------------------------------------
/**
 * Evaluate the Components from a BRDF (Smith)
 *
 * The NDF and the Fresnel term are extracted from the data.
 * This inversion requires exponentiating a matrix, so a small, self-contained
 * row major matrix API is implemented first.
 */
class matrix {
	std::vector<double> mij;
	int size;
public:
	matrix(int size);
	double& operator()(int i, int j) {return mij[j*size+i];}
	const double& operator()(int i, int j) const {return mij[j*size+i];}
	void transform(const std::vector<double>& v, std::vector<double>& out) const;
	void eigenvector(std::vector<double>& out, int iterations) const;
};

matrix::matrix(int size) : mij(size * size, 0), size(size)
{}

void matrix::transform(const std::vector<double>& v, std::vector<double>& out) const
{
	out.resize(0);
	for (int j = 0; j < size; ++j) {
		out.push_back(0);
		for (int i = 0; i < size; ++i) {
			out[j]+= (*this)(i, j) * v[i];
		}
	}
}

void matrix::eigenvector(std::vector<double>& out, int iterations) const
{
	int j = 0;
	std::vector<double> vec[2];
	vec[0].resize(size);
	vec[1].resize(size);
	for (int i = 0; i < size; ++i)
		vec[j][i] = 1.0;
	for (int i = 0; i < iterations; ++i) {
		transform(vec[j], vec[1-j]);
		j = 1 - j;
	}
	out = vec[j]; // TODO the memcpy can be avoided by using references
}

void tabular::compute_p22_smith(const brdf& brdf, int res)
{
	int cnt = res - 1;
	double dtheta = sqrt(M_PI * 0.5) / (double)cnt;
	matrix km(cnt);

	for (int i = 0; i < cnt; ++i) {
		double tmp = (double)i / (double)cnt;
		double theta = tmp * sqrt(M_PI * 0.5);
		double theta_o = theta * theta;
		double cos_theta_o = cos(theta_o);
		double tan_theta_o = tan(theta_o);
		vec3 fr = brdf.eval(dir(theta_o, 0.0), dir(theta_o, 0.0));
		double fr_i = fr.intensity();
		double kji_tmp = (dtheta * pow(cos_theta_o, 6.0)) * (8.0 * fr_i);

		for (int j = 0; j < cnt; ++j) {
			const double dphi_h = M_PI / 180.0;
			double tmp = (double)j / (double)cnt;
			double theta = tmp * sqrt(M_PI * 0.5);
			double theta_h = theta * theta;
			double cos_theta_h = cos(theta_h);
			double tan_theta_h = tan(theta_h);
			double tan_product = tan_theta_h * tan_theta_o;
			double nint = 0.0;

			for (double phi_h = 0.0; phi_h < 2.0 * M_PI; phi_h+= dphi_h)
				nint+= max(1.0, tan_product * cos(phi_h));
			nint*= dphi_h;

			km(j, i) = theta * kji_tmp * nint * tan_theta_h
			         / (cos_theta_h * cos_theta_h);
		}
	}

	// compute slope pdf
	km.eigenvector(m_p22, 4);
	m_p22.push_back(0);
}

void tabular_anisotropic::compute_p22_smith(const brdf& brdf)
{
	int w = m_elevation_res - 1;
	int h = m_azimuthal_res;
	double dtheta = sqrt(M_PI * 0.5) / (double)w;
	double dphi = 2.0 * M_PI / (double)h;
	std::vector<double> kv(w * h);
	matrix km(w * h);

	// compute kernel matrix terms
	// compute Smith matrix
	for (int i2 = 0; i2 < h; ++i2)
	for (int i1 = 0; i1 < w; ++i1) {
		double tmp1 = (double)i1 / (double)w; // in [0, 1)
		double tmp2 = (double)i2 / (double)h; // in [0, 1)
		double theta = tmp1 * 0.5 * M_PI; // in [0, pi/2)
		double phi = tmp2 * 2.0 * M_PI; // in [0, 2 pi)
		double sin_theta = sin(theta);
		double zo = cos(theta);
		double xo = sin_theta * cos(phi);
		double yo = sin_theta * sin(phi);
		vec3 fr = brdf.eval(dir(theta, phi), dir(theta, phi));
		double fr_i = fr.intensity();
		double kji_tmp1 = (dtheta * dphi) * (4.0 * fr_i * pow(zo, 5.0));

		for (int j2 = 0; j2 < h; ++j2)
		for (int j1 = 0; j1 < w; ++j1) {
			double tmp1 = (double)j1 / (double)w; // in [0, 1)
			double tmp2 = (double)j2 / (double)h; // in [0, 1)
			double theta = tmp1 * 0.5 * M_PI; // in [0, pi/2)
			double phi = tmp2 * 2.0 * M_PI; // in [0, 2 pi)
			double cos_theta = cos(theta);
			double tan_theta = tan(theta);
			double slope1 = -tan_theta * cos(phi); // x slope
			double slope2 = -tan_theta * sin(phi); // y slope
			double m_dot_o = zo - xo * slope1 - yo * slope2;
			double den = cos_theta * cos_theta;
			int j = j2 * w + j1;
			int i = i2 * w + i1;
			double kji_tmp2 = tan_theta * max(0.0, m_dot_o) / den;

			km(j, i) = kji_tmp1 * kji_tmp2;
		}
	}

	// compute slope pdf
	km.eigenvector(kv, 4);

	// compute pdf
	for (int j = 0; j < h; ++j) {
		for (int i = 0; i < w; ++i) {
			double pdf = kv[j * w + i];
			m_p22.push_back(pdf);
		}
		m_p22.push_back(0.0);
	}
}

// -------------------------------------------------------------------------------------------------
// Fresnel computation
void tabular::compute_fresnel(const brdf& brdf, int res)
{
	std::vector<vec3> fresnel(res);
	int cnt = res - 1;

	// compute average ratio between input and Torrance Sparrow equation
	for (int i = 0; i < cnt; ++i) {
		const double phi_d = M_PI * 0.5; // this can NOT be tweaked
		const double phi_h = 0.0;        // this can be tweaked (no impact on MERL)
		double tmp = (double)i / (double)cnt;
		double theta_d = tmp * M_PI * 0.5; // linear parameterization in theta_d
		vec3 f = vec3(0); // Fresnel value at R, G, B wavelengths
		int count[3] = {0, 0, 0};
		double theta_h = 0.0;

		for (int j = 0; theta_h < M_PI * 0.5 - theta_d; ++j) {
			double tmp1 = (double)j / (double)cnt;
			theta_h = tmp1 * tmp1 * M_PI * 0.5; // in [0, pi/2)

			if (theta_h > M_PI * 0.5) continue;

			dir dir_h(theta_h, phi_h);
			dir dir_d(theta_d, phi_d);
			dir dir_i, dir_o;
			half_diff_coords_to_std(dir_h, dir_d, dir_i, dir_o);

			vec3 fr1 = brdf.eval(dir_i, dir_o);
			vec3 fr2 = eval(dir_i, dir_o);

			if (fr2.x > /* epsilon */1e-4) {
				double ratio = fr1.x / fr2.x;
				f.x+= ratio;
				++count[0];
			}
			if (fr2.y > /* epsilon */1e-4) {
				double ratio = fr1.y / fr2.y;
				f.y+= ratio;
				++count[1];
			}
			if (fr2.z > /* epsilon */1e-4) {
				double ratio = fr1.z / fr2.z;
				f.z+= ratio;
				++count[2];
			}
		}

		// compute average
		fresnel[i].x = count[0] == 0 ? 1.0 : min(1.0, f.x / (double)count[0]);
		fresnel[i].y = count[1] == 0 ? 1.0 : min(1.0, f.y / (double)count[1]);
		fresnel[i].z = count[2] == 0 ? 1.0 : min(1.0, f.z / (double)count[2]);
	}
	// copy last value
	fresnel[res - 1] = fresnel[res - 2];
	set_fresnel(fresnel::spline(fresnel));
#ifndef NVERBOSE
	DJB_LOG("djb_verbose: Fresnel function ready\n");
#endif
}

void tabular_anisotropic::compute_fresnel(const brdf& brdf, int res)
{
	std::vector<vec3> fresnel(res);
	int cnt = res - 1;

	// compute average ratio between input and Torrance Sparrow equation
	for (int i = 0; i < cnt; ++i) {
		const double phi_d = M_PI * 0.5; // this can NOT be tweaked
		const double phi_h = 0.0;        // this can be tweaked (no impact on MERL)
		double tmp = (double)i / (double)cnt;
		double theta_d = tmp * M_PI * 0.5; // linear parameterization in theta_d
		vec3 f = vec3(0); // Fresnel value at R, G, B wavelengths
		int count[3] = {0, 0, 0};
		double theta_h = 0.0;

		for (int j = 0; theta_h < M_PI * 0.5 - theta_d; ++j) {
			double tmp1 = (double)j / (double)cnt;
			theta_h = tmp1 * tmp1 * M_PI * 0.5; // in [0, pi/2)

			if (theta_h > M_PI * 0.5) continue;

			dir dir_h(theta_h, phi_h);
			dir dir_d(theta_d, phi_d);
			dir dir_i, dir_o;
			half_diff_coords_to_std(dir_h, dir_d, dir_i, dir_o);

			vec3 fr1 = brdf.eval(dir_i, dir_o);
			vec3 fr2 = eval(dir_i, dir_o);

			if (fr2.x > /* epsilon */1e-4) {
				double ratio = fr1.x / fr2.x;
				f.x+= ratio;
				++count[0];
			}
			if (fr2.y > /* epsilon */1e-4) {
				double ratio = fr1.y / fr2.y;
				f.y+= ratio;
				++count[1];
			}
			if (fr2.z > /* epsilon */1e-4) {
				double ratio = fr1.z / fr2.z;
				f.z+= ratio;
				++count[2];
			}
		}

		// compute average
		fresnel[i].x = count[0] == 0 ? 1.0 : min(1.0, f.x / (double)count[0]);
		fresnel[i].y = count[1] == 0 ? 1.0 : min(1.0, f.y / (double)count[1]);
		fresnel[i].z = count[2] == 0 ? 1.0 : min(1.0, f.z / (double)count[2]);
	}
	// copy last value
	fresnel[res - 1] = fresnel[res - 2];
	set_fresnel(fresnel::spline(fresnel));
#ifndef NVERBOSE
	DJB_LOG("djb_verbose: Fresnel function ready\n");
#endif
}

// -------------------------------------------------------------------------------------------------
// Compute the CDF (mostly used for debug)
void tabular::compute_cdf()
{
	int cnt = (int)m_p22.size() - 1;
	double dtheta = sqrt(0.5 * M_PI) / (double)cnt;

	m_cdf.resize(0);
	for (int i = 0; i < cnt; ++i) {
		double nint = 0.0;

		for (int j = 0; j <= i; ++j) {
			double tmp = (double)j / (double)cnt;
			double theta = tmp * sqrt(M_PI * 0.5);
			double theta_sqr = theta * theta; // in [0, pi/2)
			double cos_theta = cos(theta_sqr);
			double pdf = p22(dir(theta_sqr, 0));

			nint+= (theta * tan(theta_sqr) * pdf)
			     / (cos_theta * cos_theta);
		}
		nint*= (2.0 * dtheta) * /* normalize */(2.0 * M_PI);
		m_cdf.push_back(nint);
	}
	m_cdf.push_back(1);

#ifndef NVERBOSE
	DJB_LOG("djb_verbose: Slope CDF ready\n");
#endif
}

// -------------------------------------------------------------------------------------------------
// Compute the Quantile Function
void tabular::compute_qf()
{
	int cnt = (int)m_p22.size() - 1;
	int res = cnt * 8; // resolution of inversion

	m_qf.resize(0);
	m_qf.push_back(0);
	for (int i = 1; i < cnt; ++i) {
		double cdf = (double)i / (double)cnt;

		for (int j = 0; j < res; ++j) {
			double tmp = (double)j / (double)res;
			double theta = tmp * M_PI * 0.5;
			double qf = cdf_radial(theta);

			// lerp lookup
			if (qf >= cdf) {
				m_qf.push_back(tmp);
				break;
			} else if (j == res) {
				m_qf.push_back(1.0);
			}
		}
	}
	m_qf.push_back(1.0);
#ifndef NVERBOSE
	DJB_LOG("djb_verbose: Slope QF ready\n");
#endif
}

// -------------------------------------------------------------------------------------------------
// Slope space p22 fetch
double tabular_anisotropic::p22(double x, double y) const
{
	double theta = atan(sqrt(x * x + y * y));
	double phi = atan2(x, y);

	return p22(dir(theta, phi));
}

// -------------------------------------------------------------------------------------------------
// Tabular Anisotropic Fetches
double tabular_anisotropic::pdf1(double phi) const
{
	double z = phi * 0.5 / M_PI;
	while (z < 0.0) z+= 1.0;
	while (z > 1.0) z-= 1.0;

	return spline_eval(m_pdf1, z);
}

double tabular_anisotropic::cdf1(double phi) const
{
	double z = phi * 0.5 / M_PI;
	while (z < 0.0) z+= 1.0;
	while (z > 1.0) z-= 1.0;

	return spline_eval(m_cdf1, z);
}

double tabular_anisotropic::qf1(double u1) const
{
	DJB_ASSERT(u1 >= 0.0 && u1 <= 1.0 && "Invalid Variate");
	return spline_eval(m_qf1, u1);
}

double tabular_anisotropic::pdf2(double theta, double phi) const
{
	DJB_ASSERT(theta >= 0.0 && theta <= 0.5 * M_PI && "Invalid Angle");
	int w = m_elevation_res;
	int h = m_azimuthal_res;
	double x = theta * 2.0 / M_PI; // in [0, 1]
	double y = phi * 0.5 / M_PI; // in [0, 1]
	while (y < 0.0) y+= 1.0;
	while (y >= 1.0) y-= 1.0; // repeat

	return spline2d_eval(m_pdf2, w, h, x, y);
}

double tabular_anisotropic::cdf2(double theta, double phi) const
{
	DJB_ASSERT(theta >= 0.0 && theta <= 0.5 * M_PI && "Invalid Angle");
	int w = m_elevation_res;
	int h = m_azimuthal_res;
	double x = theta * 2.0 / M_PI; // in [0, 1]
	double y = phi * 0.5 / M_PI; // in [0, 1]
	while (y < 0.0) y+= 1.0;
	while (y >= 1.0) y-= 1.0; // repeat

	return spline2d_eval(m_cdf2, w, h, x, y);
}

double tabular_anisotropic::qf2(double u2, double u1) const
{
	DJB_ASSERT(u1 >= 0.0 && u1 <= 1.0 && "Invalid Variate");
	DJB_ASSERT(u2 >= 0.0 && u2 <= 1.0 && "Invalid Variate");
	int w = m_elevation_res;
	int h = m_azimuthal_res;

	return spline2d_eval(m_qf2, w, h, u2, u1);
}

// -------------------------------------------------------------------------------------------------
/**
 * Compute the Marginal PDF (stored for debug)
 *
 * We compute the marginal slope PDF in (hemi-)spherical space.
 * The elevation parameter is marginalized.
 * Spherical space is more conveniant for numerical work than slope space:
 * it is free from singularities if we marginalize the elevation parameter.
 */
void tabular_anisotropic::compute_pdf1()
{
	int ntheta = 256;
	int nphi   = m_azimuthal_res;
	double dtheta = 0.5 * M_PI / (double)ntheta;

	m_pdf1.resize(0);
	for (int i = 0; i < nphi; ++i) {
		double tmp = (double)i / (double)nphi; // in (0,1)
		double phi = tmp * 2.0 * M_PI; // in [0,2pi)
		double nint = 0.0;

		for (int j = 0; j < ntheta; ++j) {
			double tmp = (double)j / (double)ntheta; // in (0,1)
			double theta = tmp * 0.5 * M_PI; // in [0,pi/2)
			double p22 = this->p22(dir(theta, phi));
			double cos_theta = cos(theta);

			nint+= (p22 * tan(theta)) / (cos_theta * cos_theta);
		}
		nint*= dtheta;
		m_pdf1.push_back(nint);
	}
	normalize_pdf1();
#ifndef NVERBOSE
	DJB_LOG("djb_verbose: PDF_1 ready\n");
#endif
}

// -------------------------------------------------------------------------------------------------
// Compute CDF 1 (stored for debug)
void tabular_anisotropic::compute_cdf1()
{
	int cnt = (int)m_azimuthal_res - 2;
	double dphi = 2.0 * M_PI / (double)cnt;

	m_cdf1.resize(0);
	m_cdf1.push_back(0.0);
	for (int i = 0; i < cnt; ++i) {
		double nint = 0.0;

		for (int j = 0; j <= i; ++j) {
			double tmp = (double)j / (double)cnt;
			double phi = tmp * 2.0 * M_PI;
			double pdf = pdf1(phi);

			nint+= pdf;
		}
		nint*= dphi;
		m_cdf1.push_back(nint);
	}
	m_cdf1.push_back(1.0);
#ifndef NVERBOSE
	DJB_LOG("djb_verbose: CDF_1 ready\n");
#endif
}

// -------------------------------------------------------------------------------------------------
// Compute the Quantile Function
void tabular_anisotropic::compute_qf1()
{
	int cnt = (int)m_cdf1.size() - 1;
	int res = cnt * 8; // resolution of inversion

	m_qf1.resize(0);
	m_qf1.push_back(0);
	for (int i = 1; i < cnt; ++i) {
		double cdf = (double)i / (double)cnt;

		for (int j = 0; j < res; ++j) {
			double tmp = (double)j / (double)res;
			double phi = tmp * 2.0 * M_PI;
			double qf = cdf1(phi);

			// lerp lookup
			if (qf >= cdf) {
				m_qf1.push_back(tmp);
				break;
			} else if (j == res) {
				m_qf1.push_back(1.0);
			}
		}
	}
	m_qf1.push_back(1.0);
#ifndef NVERBOSE
	DJB_LOG("djb_verbose: QF_1 ready\n");
#endif
}

// -------------------------------------------------------------------------------------------------
/**
 * Compute the Conditional PDF (stored for debug)
 *
 * We compute the conditional PDF in (hemi-)spherical space.
 * This is the PDF of elevation angle conditioned on azimuthal angle.
 */
void tabular_anisotropic::compute_pdf2()
{
	int ntheta = m_elevation_res - 1;
	int nphi = m_azimuthal_res;

	m_pdf2.resize(0);
	for (int i = 0; i < nphi; ++i) {
		double tmp = (double)i / (double)nphi; // in [0,1)
		double phi = tmp * 2.0 * M_PI; // in [0,2pi)

		for (int j = 0; j < ntheta; ++j) {
			double tmp = (double)j / (double)ntheta; // in [0,1)
			double theta = tmp * 0.5 * M_PI; // in [0,pi/2)
			double p22 = this->p22(dir(theta, phi));
			double p1 = this->pdf1(phi);
			DJB_ASSERT(p1 > 0.0 && "Undefined PDF");

			m_pdf2.push_back(p22 / p1);
		}
		m_pdf2.push_back(0);
	}
	normalize_pdf2();
#ifndef NVERBOSE
	DJB_LOG("djb_verbose: PDF_2 ready\n");
#endif
}

// -------------------------------------------------------------------------------------------------
// Compute CDF 2 (stored for debug)
void tabular_anisotropic::compute_cdf2()
{
	int ntheta = m_elevation_res - 1;
	int nphi = m_azimuthal_res;
	double dtheta = 0.5 * M_PI / (double)ntheta;

	m_cdf2.resize(0);
	for (int i = 0; i < nphi; ++i) {
		double tmp = (double)i / (double)nphi; // in [0,1)
		double phi = tmp * 2.0 * M_PI; // in [0,2pi)

		for (int j = 0; j < ntheta; ++j) {
			double nint = 0.0;

			for (int k = 0; k <= j; ++k) {
				double tmp = (double)k / (double)ntheta; // in [0,1)
				double theta = tmp * 0.5 * M_PI; // in [0,pi/2)
				double pdf2 = this->pdf2(theta, phi);
				double cos_theta = cos(theta);

				nint+= (pdf2 * tan(theta)) / (cos_theta * cos_theta);
			}
			nint*= dtheta;
			m_cdf2.push_back(nint);
		}
		m_cdf2.push_back(1);
	}
#ifndef NVERBOSE
	DJB_LOG("djb_verbose: CDF_2 ready\n");
#endif
}

// -------------------------------------------------------------------------------------------------
// Compute the Quantile Function
void tabular_anisotropic::compute_qf2()
{
	int ntheta = m_elevation_res - 1;
	int nphi = m_azimuthal_res;
	int res = ntheta * 8; // resolution of inversion

	m_qf2.resize(0);
	for (int k = 0; k < nphi; ++k) {
		double tmp = (double)k / (double)nphi;
		double phi = tmp * 2.0 * M_PI;

		m_qf2.push_back(0);
		for (int i = 1; i < ntheta; ++i) {
			double cdf = (double)i / (double)ntheta;

			for (int j = 0; j < res; ++j) {
				double tmp = (double)j / (double)res;
				double theta = tmp * 0.5 * M_PI;
				double qf = cdf2(theta, phi);

				// lerp lookup
				if (qf >= cdf) {
					m_qf2.push_back(tmp);
					break;
				} else if (j == res) {
					m_qf2.push_back(1.0);
				}
			}
		}
		m_qf2.push_back(1.0);
	}
#ifndef NVERBOSE
	DJB_LOG("djb_verbose: QF_2 ready\n");
#endif
}

// -------------------------------------------------------------------------------------------------
// Normalize pdf1
void tabular_anisotropic::normalize_pdf1()
{
	int cnt = 512;
	double dphi = 2.0 * M_PI / (double)cnt;
	double nint = 0.0;
	double k;

	for (int i = 0; i < cnt; ++i) {
		double tmp = (double)i / (double)cnt; // in (0,1)
		double phi = tmp * 2.0 * M_PI; // in [0,2pi)
		double p1 = this->pdf1(phi);

		nint+= p1;
	}
	nint*= dphi;
	k = 1.0 / nint;
	for (int i = 0; i < (int)m_pdf1.size(); ++i)
		m_pdf1[i]*= k;
#ifndef NVERBOSE 
	DJB_LOG("djb_verbose: PDF P1 norm. cst: %.9f\n", k);
#endif
}

// -------------------------------------------------------------------------------------------------
// Normalize pdf2
void tabular_anisotropic::normalize_pdf2()
{
	int ntheta = 256;
	int nphi = m_azimuthal_res;
	double dtheta = 0.5 * M_PI / (double)ntheta;
	std::vector<double> k(nphi, 0);

	k.resize(0);
	for (int j = 0; j < nphi; ++j) {
		double tmp = (double)j / (double)nphi;
		double phi = tmp * 2.0 * M_PI; // in [0,2pi)
		double nint = 0.0;

		for (int i = 0; i < ntheta; ++i) {
			double tmp = (double)i / (double)ntheta; // in [0,1)
			double theta = tmp * 0.5 * M_PI; // in [0,pi/2)
			double pdf2 = this->pdf2(theta, phi);
			double cos_theta = cos(theta);

			nint+= (pdf2 * tan(theta)) / (cos_theta * cos_theta);
		}
		nint*= dtheta;
		k.push_back(1.0 / nint);
	}

	for (int j = 0; j < nphi; ++j) {
		for (int i = 0; i < m_elevation_res; ++i)
			m_pdf2[i + m_elevation_res * j]*= k[j];
#ifndef NVERBOSE 
		DJB_LOG("djb_verbose: PDF P2 norm. cst: %.9f\n", k[j]);
#endif
	}
}

// -------------------------------------------------------------------------------------------------
// Sample + PDF
dir tabular_anisotropic::sample(const dir& out, double u1, double u2) const
{
	double tmp = qf1(u2);
	double phi = tmp * 2.0 * M_PI;
	double theta = qf2(u1, tmp) * 0.5 * M_PI;

	return dir(theta, phi);
}

double tabular_anisotropic::pdf(const dir& out, const dir& m) const
{
	return p22(m) / pow(cos(m.theta), 3.0);
}

// -------------------------------------------------------------------------------------------------
// Accessors
const std::vector<double>&
tabular_anisotropic::get_p22v(int *elevcnt, int *azimcnt) const
{
	if (elevcnt) *elevcnt = m_elevation_res;
	if (azimcnt) *azimcnt = m_azimuthal_res;

	return m_p22;
}
const std::vector<double>&
tabular_anisotropic::get_g1v(int *elevcnt, int *azimcnt) const
{
	if (elevcnt) *elevcnt = m_elevation_res;
	if (azimcnt) *azimcnt = m_azimuthal_res;

	return m_g1;
}

// -------------------------------------------------------------------------------------------------
/**
 * Direct Conversion to Parametric BRDFs
 *
 * The tabulated BRDFs may be converted directly to either a 
 * Gaussian or a GGX BRDF. For this, moments are computed from the 
 * tabulated microfacet slope distribution in order to retrieve
 * the scale and correlation parameters.
 */
gaussian *tabular::to_gaussian(const tabular& tab)
{
	const int ntheta = 128;
	double dtheta = sqrt(M_PI * 0.5) / (double)ntheta;
	double nint = 0.0;
	double sigma;

	for (int i = 0; i < ntheta; ++i) {
		double tmp = (double)i / (double)ntheta; // in [0,1)
		double theta = tmp * sqrt(M_PI * 0.5);
		double theta_sqr = theta * theta; // in [0, pi/2)
		double c = cos(theta_sqr);
		double pdf = tab.p22(dir(theta_sqr, 0.0));
		double tan_theta_sqr = tan(theta_sqr); // first moment
		double moment = tan_theta_sqr * tan_theta_sqr; // second moment

		nint+= (moment * theta) * (pdf * tan_theta_sqr) / (c * c);
	}
	nint*= (2.0 * dtheta) * /* int_0^2pi cos^2 phi dphi */M_PI;
	sigma = sqrt(nint);

#ifndef NVERBOSE
	DJB_LOG("djb_verbose: Gaussian_sigma = %.9f\n", sigma);
#endif

	return new gaussian(tab.m_gaf, sigma, *tab.m_fresnel, tab.m_shadow);
}

ggx *tabular::to_ggx(const tabular& tab)
{
	const int ntheta = 128;
	double dtheta = sqrt(M_PI * 0.5) / (double)ntheta;
	double nint = 0.0;
	double sigma;

	for (int i = 0; i < ntheta; ++i) {
		double tmp = (double)i / (double)ntheta; // in [0,1)
		double theta = tmp * sqrt(M_PI * 0.5);
		double theta_sqr = theta * theta; // in [0, pi/2)
		double c = cos(theta_sqr);
		double pdf = tab.p22(dir(theta_sqr, 0.0));
		double tan_theta_sqr = tan(theta_sqr); // first moment
		double moment = fabs(tan_theta_sqr); // second moment

		nint+= (moment * theta) * (pdf * tan_theta_sqr) / (c * c);
	}
	nint*= (2.0 * dtheta) * /* int_0^2pi fabs(cos phi) dphi */4.0;
	sigma = nint;
#ifndef NVERBOSE
	DJB_LOG("djb_verbose: GGX_sigma = %.9f\n", sigma);
#endif

	return new ggx(tab.m_gaf, sigma, *tab.m_fresnel, tab.m_shadow);
}

// -------------------------------------------------------------------------------------------------
/**
 * Direct Conversion to Parametric BRDFs
 *
 * The tabulated BRDFs may be converted directly to either a 
 * Gaussian or a GGX BRDF. For this, moments are computed from the 
 * tabulated microfacet slope distribution in order to retrieve
 * the scale and correlation parameters.
 */
gaussian *tabular_anisotropic::to_gaussian(const tabular_anisotropic& tab)
{
	const int ntheta = 128;
	const int nphi   = 512;
	double dtheta = sqrt(M_PI * 0.5) / (double)ntheta;
	double dphi = 2.0 * M_PI / (double)nphi;
	double nint[3] = {0.0, 0.0, 0.0};
	double r1, r2, rangle;

	for (int j = 0; j < nphi; ++j) {
		double tmp = (double)j / (double)nphi; // in [0,1)
		double phi = tmp * 2.0 * M_PI; // in [0,2pi)
		double cos_phi = cos(phi);
		double sin_phi = sin(phi);
		double cos_phi_sqr = cos_phi * cos_phi;
		double sin_phi_sqr = sin_phi * sin_phi;

		for (int i = 0; i < ntheta; ++i) {
			double tmp1 = (double)i / (double)ntheta; // in [0,1)
			double theta = tmp1 * sqrt(M_PI * 0.5); // in [0,sqrt(pi/2))
			double theta_sqr = theta * theta; // in [0, pi/2)
			double p22 = tab.p22(dir(theta_sqr, phi));
			double tan_theta = tan(theta_sqr);
			double cos_theta = cos(theta_sqr);
			double tan_theta_sqr = tan_theta * tan_theta;
			double cos_theta_sqr = cos_theta * cos_theta;
			double tmp2 = theta * p22 * tan_theta / cos_theta_sqr;
			double e11 = tan_theta_sqr * cos_phi_sqr; // var1
			double e22 = tan_theta_sqr * sin_phi_sqr; // var2
			double e12 = tan_theta_sqr * cos_phi * sin_phi; // cov

			nint[0]+= tmp2 * e11;
			nint[1]+= tmp2 * e22;
			nint[2]+= tmp2 * e12;
		}
	}
	for (int i = 0; i < 3; i++)
		nint[i]*= 2.0 * dtheta * dphi;
	covariance_to_ellipse(nint[0], nint[1], nint[2], &r1, &r2, &rangle);

#ifndef NVERBOSE
	DJB_LOG("djb_verbose: Gaussian_params = %.3f %.3f %.3f\n", r1, r2, rangle);
#endif

	return new gaussian(tab.m_gaf, r1, r2, rangle, *tab.m_fresnel, tab.m_shadow);
}

ggx *tabular_anisotropic::to_ggx(const tabular_anisotropic& tab)
{
	const int ntheta = 128;
	const int nphi   = 512;
	double dtheta = sqrt(M_PI * 0.5) / (double)ntheta;
	double dphi = 2.0 * M_PI / (double)nphi;
	double nint[4] = {0.0, 0.0, 0.0, 0.0};
	double var1, var2, cov;
	double r1, r2, rangle;

	for (int j = 0; j < nphi; ++j) {
		double tmp = (double)j / (double)nphi; // in [0,1)
		double phi = tmp * 2.0 * M_PI; // in [0,2pi)
		double cos_phi = cos(phi);
		double sin_phi = sin(phi);

		for (int i = 0; i < ntheta; ++i) {
			double tmp1 = (double)i / (double)ntheta; // in [0,1)
			double theta = tmp1 * sqrt(M_PI * 0.5); // in [0,sqrt(pi/2))
			double theta_sqr = theta * theta; // in [0, pi/2)
			double p22 = tab.p22(dir(theta_sqr, phi));
			double tan_theta = tan(theta_sqr);
			double cos_theta = cos(theta_sqr);
			double cos_theta_sqr = cos_theta * cos_theta;
			double tmp2 = theta * p22 * tan_theta / cos_theta_sqr;
			double e11 = fabs(tan_theta * cos_phi); // scale1
			double e22 = fabs(tan_theta * sin_phi); // scale2
			double e12 = cos_phi * sin_phi; // co-scale
			double e21 = cos_phi * cos_phi; // co-scale

			nint[0]+= tmp2 * e11;
			nint[1]+= tmp2 * e22;
			nint[2]+= tmp2 * e12;
			nint[3]+= tmp2 * e21;
		}
	}
	for (int i = 0; i < 4; i++)
		nint[i]*= 2.0 * dtheta * dphi;
	var1 = nint[0] * nint[0];
	var2 = nint[1] * nint[1];
	cov = var1 * nint[2] / (nint[2] * nint[2] + nint[3] * nint[3]);
	covariance_to_ellipse(var1, var2, cov, &r1, &r2, &rangle);

#ifndef NVERBOSE
	DJB_LOG("djb_verbose: Gaussian_params = %.3f %.3f %.3f\n", r1, r2, rangle);
#endif

	return new ggx(tab.m_gaf, r1, r2, rangle, *tab.m_fresnel, tab.m_shadow);
}

// *************************************************************************************************
// Shifted Gamma Distribution API implementation

const sgd::data sgd::s_data[100] = {
	{ "alum-bronze", "alum-bronze", { 0.0478786, 0.0313514, 0.0200638 }, { 0.0364976, 0.664975, 0.268836 }, { 0.014832, 0.0300126, 0.0490339 }, { 0.459076, 0.450056, 0.529272 }, { 6.05524, 0.235756, 0.580647 }, { 5.05524, 0.182842, 0.476088 }, { 46.3841, 24.5961, 14.8261 }, { 2.60672, 2.97371, 2.7827 }, { 1.12717e-07, 1.06401e-07, 5.27952e-08 }, { 47.783, 36.2767, 31.6066 }, { 0.205635, 0.066289, -0.0661091 }, { 0.100735, 0.0878706, 0.0861907 } },
	{ "alumina-oxide", "alumina-oxide", { 0.316358, 0.292248, 0.25416 }, { 0.00863128, 0.00676832, 0.0103309 }, { 0.000159222, 0.000139421, 0.000117714 }, { 0.377727, 0.318496, 0.402598 }, { 0.0300766, 1.70375, 1.96622 }, { -0.713784, 0.70375, 1.16019 }, { 4381.96, 5413.74, 5710.42 }, { 3.31076, 4.93831, 2.84538 }, { 6.72897e-08, 1.15769e-07, 6.32199e-08 }, { 354.275, 367.448, 414.581 }, { 0.52701, 0.531166, 0.53301 }, { 0.213276, 0.147418, 0.27746 } },
	{ "aluminium", "aluminium", { 0.0305166, 0.0358788, 0.0363463 }, { 0.0999739, 0.131797, 0.0830361 }, { 0.0012241, 0.000926487, 0.000991844 }, { 0.537669, 0.474562, 0.435936 }, { 0.977854, 0.503108, 1.77905 }, { -0.0221457, -0.0995445, 0.77905 }, { 449.321, 658.044, 653.86 }, { 8.2832e-07, 9.94692e-08, 6.11887e-08 }, { 3.54592e-07, 16.0175, 15.88 }, { 23.8656, 10.6911, 9.69801 }, { -0.510356, 0.570179, 0.566156 }, { 0.303567, 0.232628, 0.441578 } },
	{ "aventurnine", "aventurnine", { 0.0548217, 0.0621179, 0.0537826 }, { 0.0348169, 0.0872381, 0.111961 }, { 0.000328039, 0.000856166, 0.00145342 }, { 0.387167, 0.504525, 0.652122 }, { 0.252033, 0.133897, 0.087172 }, { 0.130593, 0.0930416, 0.0567429 }, { 2104.51, 676.157, 303.59 }, { 3.12126, 2.50965e-07, 2.45778e-05 }, { 1.03849e-07, 8.53824e-07, 3.20722e-07 }, { 251.265, 24.2886, 29.0236 }, { 0.510125, -0.41764, -0.245097 }, { 0.0359759, 0.0297523, 0.0285881 } },
	{ "beige-fabric", "fabric-beige", { 0.20926, 0.160666, 0.145337 }, { 0.121663, 0.0501577, 0.00177279 }, { 0.39455, 0.15975, 0.110706 }, { 0.474725, 0.0144728, 1.70871e-12 }, { 0.0559459, 0.222268, 8.4764 }, { -0.318718, -0.023826, 7.4764 }, { 3.8249, 7.32453, 10.0904 }, { 2.26283, 2.97144, 3.55311 }, { 0.0375346, 0.073481, 0.0740222 }, { 7.52635, 9.05672, 10.6185 }, { 0.217453, 0.407084, 0.450203 }, { 0.00217528, 0.00195262, 0.00171008 } },
	{ "black-fabric", "fabric-black", { 0.0189017, 0.0112353, 0.0110067 }, { 2.20654e-16, 6.76197e-15, 1.57011e-13 }, { 0.132262, 0.128044, 0.127838 }, { 0.189024, 0.18842, 0.188426 }, { 1, 1, 1 }, { 0, 0, 0 }, { 8.1593, 8.38075, 8.39184 }, { 3.83017, 3.89536, 3.89874 }, { 0.00415117, 0.00368324, 0.00365826 }, { 12.9974, 13.2597, 13.2737 }, { 0.207997, 0.205597, 0.205424 }, { 0.000363154, 0.000272253, 0.000274773 } },
	{ "black-obsidian", "black-obsidian", { 0.00130399, 0.0011376, 0.00107233 }, { 0.133029, 0.125362, 0.126188 }, { 0.000153649, 0.000148939, 0.000179285 }, { 0.186234, 0.227495, 0.25745 }, { 2.42486e-12, 0.0174133, 0.091766 }, { -0.0800755, -0.048671, 0.0406445 }, { 5668.57, 5617.79, 4522.84 }, { 13.7614, 8.59526, 6.44667 }, { 1e+38, 1e+38, 1e+38 }, { 117.224, 120.912, 113.366 }, { 1.19829, 1.19885, 1.19248 }, { 0.0841374, 0.0943085, 0.123514 } },
	{ "black-oxidized-steel", "black-oxidized-steel", { 0.0149963, 0.0120489, 0.0102471 }, { 0.373438, 0.344382, 0.329202 }, { 0.187621, 0.195704, 0.200503 }, { 0.661367, 0.706913, 0.772267 }, { 0.0794166, 0.086518, 0.080815 }, { 0.0470402, 0.0517633, 0.0455037 }, { 5.1496, 4.91636, 4.69009 }, { 4.0681, 3.95489, 3.71052 }, { 1.07364e-07, 1.05341e-07, 1.16556e-07 }, { 20.2383, 20.1786, 20.2553 }, { -0.479617, -0.4885, -0.478388 }, { 0.00129576, 0.0011378, 0.000986163 } },
	{ "black-phenolic", "black-phenolic", { 0.00204717, 0.00196935, 0.00182908 }, { 0.177761, 0.293146, 0.230592 }, { 0.00670804, 0.00652009, 0.00656043 }, { 0.706648, 0.677776, 0.673986 }, { 0.16777, 0.12335, 0.166663 }, { 0.111447, 0.0927321, 0.125663 }, { 65.4189, 70.8936, 70.9951 }, { 1.06318, 1.15283, 1.16529 }, { 1.24286e-07, 3.00039e-08, 9.77334e-08 }, { 74.0711, 75.1165, 73.792 }, { 0.338204, 0.319306, 0.33434 }, { 0.0307129, 0.0531183, 0.0454238 } },
	{ "black-soft-plastic", "black-plastic-soft", { 0.00820133, 0.00777718, 0.00764537 }, { 0.110657, 0.0980322, 0.100579 }, { 0.0926904, 0.0935964, 0.0949975 }, { 0.14163, 0.148703, 0.143694 }, { 0.150251, 0.169418, 0.170457 }, { 0.100065, 0.113089, 0.114468 }, { 11.2419, 11.113, 10.993 }, { 4.3545, 4.3655, 4.31586 }, { 0.00464641, 0.00384785, 0.0046145 }, { 14.6751, 14.8089, 14.5436 }, { 0.275651, 0.262317, 0.271284 }, { 0.000527402, 0.000550255, 0.000527718 } },
	{ "blue-acrylic", "acrylic-blue", { 0.0134885, 0.0373766, 0.10539 }, { 0.0864901, 0.0228191, 0.204042 }, { 0.000174482, 0.000269795, 0.0015211 }, { 0.373948, 0.362425, 0.563636 }, { 0.0185562, 0.399982, 0.0525861 }, { -0.0209713, 0.241543, 0.0169474 }, { 4021.24, 2646.36, 346.898 }, { 3.38722, 3.62885, 1.83684e-06 }, { 9.64334e-08, 9.96105e-08, 3.61787e-07 }, { 338.073, 272.828, 23.5039 }, { 0.526039, 0.515404, -0.526935 }, { 0.0612235, 0.0789826, 0.0461093 } },
	{ "blue-fabric", "fabric-blue", { 0.0267828, 0.0281546, 0.066668 }, { 0.0825614, 0.0853369, 0.0495164 }, { 0.248706, 0.249248, 0.18736 }, { 9.23066e-13, 1.66486e-12, 2.27218e-12 }, { 0.201626, 0.213723, 0.56548 }, { 0.225891, 0.226267, 0.638493 }, { 5.15615, 5.14773, 6.43713 }, { 2.25846, 2.25536, 2.68382 }, { 0.128037, 0.128363, 0.0944915 }, { 6.95531, 6.94633, 8.17665 }, { 0.407528, 0.407534, 0.411378 }, { 0.000323043, 0.00032064, 0.000653631 } },
	{ "blue-metallic-paint2", "ch-ball-blue-metallic", { 0.010143, 0.0157349, 0.0262717 }, { 0.0795798, 0.0234493, 0.0492337 }, { 0.00149045, 0.00110477, 0.00141008 }, { 0.624615, 0.598721, 0.67116 }, { 9.36434e-14, 3.61858e-15, 1.15633e-14 }, { -0.210234, -1, -1 }, { 314.024, 441.812, 299.726 }, { 1.20935e-05, 7.51792e-06, 3.86474e-05 }, { 3.38901e-07, 2.94502e-07, 3.15718e-07 }, { 27.0491, 28.576, 30.6214 }, { -0.326593, -0.274443, -0.187842 }, { 0.0908879, 0.163236, 0.286541 } },
	{ "blue-metallic-paint", "metallic-blue", { 0.00390446, 0.00337319, 0.00848198 }, { 0.0706771, 0.0415082, 0.104423 }, { 0.155564, 0.139, 0.15088 }, { 1.01719, 1.02602, 1.16153 }, { 0.149347, 0.153181, 1.87241e-14 }, { -0.487331, -0.76557, -1 }, { 4.4222, 4.59265, 3.93929 }, { 2.54345, 2.33884, 2.24405 }, { 6.04906e-08, 5.81858e-08, 1.2419e-07 }, { 23.9533, 25.0641, 24.6856 }, { -0.34053, -0.294595, -0.258117 }, { 0.00202544, 0.00246268, 0.0059725 } },
	{ "blue-rubber", "blue-rubber", { 0.0371302, 0.0732915, 0.146637 }, { 0.384232, 0.412357, 0.612608 }, { 0.218197, 0.2668, 0.478375 }, { 0.815054, 1.00146, 1.24995 }, { 0.0631713, 0.0622636, 0.0399196 }, { 0.0478254, 0.0422186, 0.007015 }, { 4.41586, 3.76795, 3.46276 }, { 3.77807, 3.82679, 3.33186 }, { 1.2941e-07, 1.07194e-07, 0.00045665 }, { 19.8046, 19.3115, 11.4364 }, { -0.499472, -0.557706, -0.172177 }, { 0.000630461, 0.000733287, 0.00171092 } },
	{ "brass", "brass", { 0.0301974, 0.0223812, 0.0139381 }, { 0.0557826, 0.0376687, 0.0775998 }, { 0.0002028, 0.000258468, 0.00096108 }, { 0.362322, 0.401593, 0.776606 }, { 0.639886, 0.12354, 0.0197853 }, { -0.360114, -0.87646, -0.0919344 }, { 3517.61, 2612.49, 331.815 }, { 3.64061, 2.87206, 0.529487 }, { 1.01146e-07, 9.83073e-08, 5.48819e-08 }, { 312.802, 283.431, 183.091 }, { 0.522711, 0.516719, 0.474834 }, { 0.440765, 0.200948, 0.15549 } },
	{ "cherry-235", "cherry-235", { 0.0497502, 0.0211902, 0.0120688 }, { 0.166001, 0.202786, 0.165189 }, { 0.0182605, 0.0277997, 0.0255721 }, { 0.0358348, 0.163231, 0.129135 }, { 0.0713408, 0.0571719, 0.0791809 }, { 0.0200814, 0.00887306, 0.0251675 }, { 54.7448, 33.7294, 37.36 }, { 6.11314, 6.23697, 6.16351 }, { 30.3886, 0.00191869, 0.0495069 }, { 18.8114, 25.8454, 23.3753 }, { 0.816378, 0.387479, 0.522125 }, { 0.00235908, 0.00199931, 0.00155902 } },
	{ "chrome", "chrome", { 0.00697189, 0.00655268, 0.0101854 }, { 0.0930656, 0.041946, 0.104558 }, { 0.00013, 0.0002, 6e-5 }, { 0.3, 0.36, 0.26 }, { 0.256314, 0.819565, 3.22085e-13 }, { -0.743686, -0.180435, -1 }, { 2200, 2150, 4100 }, { 3.8545, 4.44817, 5.40959 }, { 5.30781e-08, 1.04045e-07, 1.8e+10 }, { 354.965, 349.356, 457 }, { 0.526796, 0.528469, 1.00293 }, { 0.802138, 1.29121, 1.06148 } },
	{ "chrome-steel", "chrome-steel", { 0.0206718, 0.0240818, 0.024351 }, { 0.129782, 0.109032, 0.0524555 }, { 5.51292e-05, 3.13288e-05, 4.51944e-05 }, { 0.207979, 0.152758, 0.325431 }, { 1.18818e-12, 2.06813e-11, 0.580895 }, { -0.316807, -0.265326, -0.419105 }, { 15466.8, 28628.9, 16531.2 }, { 12.8988, 68.7898, 4.68237 }, { 1e+10, 1e+10, 44.7025 }, { 510, 665, 618.155 }, { 1.20035, 1.2003, 0.579562 }, { 0.552277, 0.448267, 1.12869 } },
	{ "colonial-maple-223", "colonial-maple-223", { 0.100723, 0.0356306, 0.0162408 }, { 0.059097, 0.0661341, 0.11024 }, { 0.0197628, 0.0279336, 0.0621265 }, { 0.0311867, 0.112022, 0.344348 }, { 0.0576683, 0.0617498, 0.0479061 }, { -0.0503364, -0.0382196, -0.0382636 }, { 50.7952, 34.6835, 14.201 }, { 6.03342, 6.01053, 4.56588 }, { 18.9034, 0.087246, 1.03757e-07 }, { 18.3749, 21.6159, 26.9347 }, { 0.801289, 0.544191, -0.139944 }, { 0.00229125, 0.00193164, 0.0013723 } },
	{ "color-changing-paint1", "", { 0.00513496, 0.00500415, 0.00296872 }, { 1.53167, 0.430731, 1.48308 }, { 0.00320129, 0.0023053, 0.0329464 }, { 0.167301, 0.100003, 1.04868 }, { 0.0208525, 0.109366, 0.0553211 }, { 0.0101418, 0.0801275, 0.0357004 }, { 279.058, 407.716, 9.71061 }, { 6.73768, 7.19333, 1.04729 }, { 2.66398e+08, 8.98323e+10, 4.16363e-08 }, { 36.7257, 27.8164, 45.5388 }, { 1.01208, 1.19549, 0.131148 }, { 0.164144, 0.0515527, 0.116602 } },
	{ "color-changing-paint2", "", { 0.00463528, 0.00544054, 0.0070818 }, { 1.35172, 1.47838, 1.29831 }, { 0.0279961, 0.0267135, 0.0257468 }, { 0.720154, 0.717648, 0.694662 }, { 0.019073, 0.00825302, 0.0301024 }, { -0.0181694, -0.0198592, -0.000522292 }, { 18.6722, 19.4709, 20.7514 }, { 1.56832, 1.554, 1.60402 }, { 1.19918e-07, 9.95358e-08, 9.69838e-08 }, { 41.295, 42.1716, 42.3986 }, { 0.124794, 0.129237, 0.132024 }, { 0.0489296, 0.0499662, 0.054555 } },
	{ "color-changing-paint3", "", { 0.00305737, 0.00257341, 0.00263616 }, { 0.880793, 0.691268, 0.707821 }, { 0.014742, 0.0135513, 0.0108894 }, { 0.537248, 0.572188, 0.457665 }, { 0.055479, 0.0666783, 0.0585094 }, { 0.0315282, 0.0457554, 0.0365558 }, { 42.1798, 43.3684, 62.0624 }, { 2.06002, 1.82703, 2.54086 }, { 5.0531e-08, 9.99088e-08, 5.726e-08 }, { 50.0948, 52.0344, 54.643 }, { 0.197971, 0.229219, 0.241049 }, { 0.0484699, 0.0342594, 0.0347198 } },
	{ "dark-blue-paint", "dark-blue-paint", { 0.00665057, 0.0139696, 0.0472605 }, { 0.231099, 0.18931, 0.12528 }, { 0.130681, 0.112103, 0.0629285 }, { 0.238562, 0.17, 0.0157371 }, { 0.112486, 0.106018, 0.139316 }, { 0.0662142, 0.0682314, 0.10607 }, { 8.0828, 9.4312, 16.8231 }, { 4.05526, 4.09607, 4.54368 }, { 0.00104743, 0.0035546, 0.10813 }, { 14.347, 13.928, 13.5626 }, { 0.11894, 0.226752, 0.524826 }, { 0.000406243, 0.000291986, 0.000555778 } },
	{ "dark-red-paint", "dark-red-paint", { 0.237125, 0.0365577, 0.0106149 }, { 0.227405, 0.111055, 0.150433 }, { 0.67372, 0.0706716, 0.20328 }, { 1.38127, 1.4345e-13, 0.27723 }, { 1.56101e-11, 0.0779523, 0.136566 }, { -0.148315, 0.0560349, 0.0961999 }, { 3.95965, 15.1861, 5.71355 }, { 2.27942, 4.30359, 3.20838 }, { 0.0225467, 0.113039, 0.00667899 }, { 7.45998, 12.679, 10.9071 }, { 0.119045, 0.521154, 0.166378 }, { 0.00155397, 0.000508377, 0.000218865 } },
	{ "dark-specular-fabric", "", { 0.0197229, 0.00949167, 0.00798414 }, { 0.556218, 0.401495, 0.378651 }, { 0.140344, 0.106541, 0.166715 }, { 0.249059, 0.177611, 0.434167 }, { 0.0351133, 0.0387177, 0.0370533 }, { 0.0243153, 0.0293178, 0.0264913 }, { 7.60492, 9.81673, 6.19307 }, { 3.93869, 4.23097, 4.3775 }, { 0.00122421, 0.00238545, 8.47126e-06 }, { 13.889, 14.5743, 17.2049 }, { 0.114655, 0.210179, -0.227628 }, { 0.00158681, 0.000974676, 0.000638865 } },
	{ "delrin", "delrin", { 0.272703, 0.249805, 0.220642 }, { 0.536593, 0.727886, 0.64011 }, { 0.176535, 0.344018, 0.208011 }, { 0.762088, 0.823603, 0.85976 }, { 0.0398465, 0.0430719, 0.0290162 }, { -0.0121332, -0.0255915, -0.0305128 }, { 5.02017, 3.78669, 4.39454 }, { 3.44325, 3.49289, 3.48886 }, { 9.20654e-08, 0.000215686, 1.06338e-07 }, { 21.2534, 12.5556, 20.455 }, { -0.438862, -0.184248, -0.478699 }, { 0.00493658, 0.00655617, 0.00422698 } },
	{ "fruitwood-241", "fruitwood-241", { 0.0580445, 0.0428667, 0.0259801 }, { 0.203894, 0.233494, 0.263882 }, { 0.00824986, 0.0534794, 0.0472951 }, { 0.160382, 1.07206, 0.768335 }, { 0.00129482, 0.00689891, 0.01274 }, { -0.0211778, -0.0140791, -0.00665974 }, { 110.054, 6.98485, 11.6203 }, { 6.74678, 1.31986, 1.76021 }, { 162.121, 1.18843e-07, 1.0331e-07 }, { 32.6966, 36.752, 34.371 }, { 0.765181, 0.0514459, 0.0106852 }, { 0.00493613, 0.00460352, 0.00373027 } },
	{ "gold-metallic-paint2", "ch-ball-gold-metallic2", { 0.0796008, 0.0538361, 0.0649523 }, { 0.633627, 1.77116, 0.0564028 }, { 0.00376608, 0.00871206, 0.000572055 }, { 0.415684, 0.368424, 0.623038 }, { 0.0343265, 0.00330259, 4.15759e-12 }, { -0.00929705, -0.0219437, -0.0596944 }, { 181.769, 85.527, 795.042 }, { 2.77279, 3.50114, 0.985212 }, { 4.78294e-08, 9.8399e-08, 4.45358e-08 }, { 83.871, 57.2291, 216.537 }, { 0.365256, 0.276734, 0.491264 }, { 0.117145, 0.17457, 0.0754722 } },
	{ "gold-metallic-paint3", "ch-ball-gold-metallic", { 0.0579212, 0.0416649, 0.0271208 }, { 0.0729896, 0.0597695, 0.037684 }, { 0.00146432, 0.00156513, 0.000977438 }, { 0.529437, 0.551234, 0.504486 }, { 1.84643e-14, 5.7212e-15, 8.68546e-13 }, { -1, -1, -0.648897 }, { 382.887, 345.197, 593.725 }, { 4.97078e-07, 1.12888e-06, 4.15366e-07 }, { 3.98586e-07, 3.69533e-07, 16.0596 }, { 22.0196, 22.6462, 11.7126 }, { -0.634103, -0.588056, 0.578355 }, { 0.116233, 0.0897222, 0.0711514 } },
	{ "gold-metallic-paint", "metallic-gold", { 0.0178625, 0.00995704, 0.00335044 }, { 0.17127, 0.120714, 0.115473 }, { 0.127954, 0.127825, 0.109623 }, { 0.781093, 0.795517, 0.661313 }, { 1.00776e-12, 1.22243e-15, 6.18432e-13 }, { -1, -1, -0.334497 }, { 5.90039, 5.83556, 7.14985 }, { 2.7548, 2.7057, 2.94348 }, { 9.46481e-08, 1.06951e-07, 1.10733e-07 }, { 23.8811, 23.9059, 24.2972 }, { -0.303345, -0.293778, -0.267019 }, { 0.0095612, 0.00483452, 0.00145599 } },
	{ "gold-paint", "gold-paint", { 0.147708, 0.0806975, 0.033172 }, { 0.160592, 0.217282, 0.236425 }, { 0.122506, 0.108069, 0.12187 }, { 0.795078, 0.637578, 0.936117 }, { 9.16095e-12, 1.81225e-12, 0.0024589 }, { -0.596835, -0.331147, -0.140729 }, { 5.98176, 7.35539, 5.29722 }, { 2.64832, 3.04253, 2.3013 }, { 9.3111e-08, 8.80143e-08, 9.65288e-08 }, { 24.3593, 24.4037, 25.3623 }, { -0.284195, -0.277297, -0.245352 }, { 0.00313716, 0.00203922, 0.00165683 } },
	{ "gray-plastic", "gray-plastic", { 0.103233, 0.104428, 0.0983734 }, { 0.494656, 0.517207, 0.52772 }, { 0.00758705, 0.00848095, 0.00887135 }, { 0.557908, 0.556548, 0.545887 }, { 0.0428175, 0.0438899, 0.0569098 }, { 0.0208304, 0.0221893, 0.0375763 }, { 75.6274, 68.3098, 66.5689 }, { 1.72111, 1.76597, 1.8406 }, { 1.06783e-07, 5.31845e-08, 6.53296e-08 }, { 65.6538, 63.2018, 61.5805 }, { 0.308907, 0.283768, 0.280211 }, { 0.0512156, 0.050844, 0.0505763 } },
	{ "grease-covered-steel", "", { 0.0196306, 0.0200926, 0.0187026 }, { 0.0433721, 0.0311621, 0.0326401 }, { 0.00019081, 0.000173919, 0.000217638 }, { 0.164569, 0.141125, 0.219421 }, { 7.12672e-13, 1.06789e-14, 1.61131e-13 }, { -1, -1, -1 }, { 4655.4, 5210.63, 3877.45 }, { 17.2827, 29.2523, 8.54352 }, { 1e+38, 1e+38, 1e+38 }, { 103.46, 105.572, 99.1097 }, { 1.20022, 1.20678, 1.19961 }, { 0.261184, 0.20789, 0.210192 } },
	{ "green-acrylic", "acrylic-green", { 0.0176527, 0.0761863, 0.0432331 }, { 0.0517555, 0.15899, 0.0754193 }, { 0.000185443, 5.19959e-05, 7.95188e-05 }, { 0.288191, 0.170979, 0.145492 }, { 0.418137, 0.0486445, 0.170328 }, { 0.330999, 0.0294954, 0.110218 }, { 4223.98, 16977.4, 11351.3 }, { 5.6545, 26.1353, 1.54193e+07 }, { 1e+38, 1e+38, 4.23355e+06 }, { 172.888, 204.112, 403.267 }, { 0.995381, 1.19417, 0.646645 }, { 0.138255, 0.168884, 0.116563 } },
	{ "green-fabric", "fabric-green", { 0.0511324, 0.0490447, 0.0577457 }, { 0.043898, 0.108081, 0.118528 }, { 0.0906425, 0.14646, 0.125546 }, { 0.199121, 0.21946, 0.130311 }, { 0.117671, 0.0797822, 0.0840896 }, { 0.107501, 0.0628391, 0.0668466 }, { 11.1681, 7.43507, 8.69777 }, { 4.65909, 3.72793, 3.72472 }, { 0.00055264, 0.00331292, 0.0119365 }, { 16.8276, 12.7802, 12.1538 }, { 0.153958, 0.17378, 0.291679 }, { 0.000577582, 0.000635885, 0.000729507 } },
	{ "green-latex", "fabric-green-latex", { 0.0885476, 0.13061, 0.0637004 }, { 0.177041, 0.16009, 0.101365 }, { 0.241826, 0.21913, 0.2567 }, { 0.175925, 0.162514, 0.326958 }, { 0.0213854, 0.0498004, 0.0677643 }, { -0.0864353, -0.0518848, -0.00668045 }, { 5.17117, 5.55867, 4.84094 }, { 2.61304, 2.75985, 2.84236 }, { 0.0417628, 0.0352726, 0.0128547 }, { 8.4496, 8.91691, 9.57998 }, { 0.296393, 0.295144, 0.18077 }, { 0.000875063, 0.00103454, 0.000568252 } },
	{ "green-metallic-paint2", "ch-ball-green-metallic", { 0.00536389, 0.0147585, 0.0072232 }, { 0.0553207, 0.0656441, 0.0608999 }, { 0.00131834, 0.00140737, 0.00171711 }, { 0.436009, 0.57969, 0.54703 }, { 0.291808, 0.194127, 0.199549 }, { 0.126008, -0.0681937, 0.0148264 }, { 493.87, 362.85, 317.92 }, { 4.96631e-08, 3.41674e-06, 8.72658e-07 }, { 15.5833, 3.60732e-07, 3.78431e-07 }, { 8.31751, 25.0069, 21.7501 }, { 0.561587, -0.431953, -0.657077 }, { 0.019819, 0.0506927, 0.0279691 } },
	{ "green-metallic-paint", "green-metallic-paint", { 0.00368935, 0.0155555, 0.022272 }, { 0.185621, 0.436002, 0.322925 }, { 0.131402, 0.146271, 0.154061 }, { 1.28366, 0.865104, 0.944013 }, { 0.12483, 0.0443223, 0.0955612 }, { 0.028874, -0.118581, -0.139373 }, { 3.62502, 5.13725, 4.71161 }, { 1.97389, 2.72437, 2.63239 }, { 5.5607e-08, 1.1239e-07, 9.76666e-08 }, { 27.5467, 23.1688, 23.2911 }, { -0.204638, -0.326555, -0.334024 }, { 0.00120994, 0.00217841, 0.00218444 } },
	{ "green-plastic", "green-plastic", { 0.015387, 0.0851675, 0.0947402 }, { 0.0607427, 0.156977, 0.125155 }, { 0.000302146, 0.00197038, 0.000690284 }, { 0.373134, 0.751741, 0.505735 }, { 0.116395, 0.0388464, 0.0476683 }, { 0.0123694, -0.0038785, 0.00864048 }, { 2329.55, 181.586, 833.886 }, { 3.39581, 0.000128411, 3.06753e-07 }, { 6.38801e-08, 2.49751e-07, 5.82332e-07 }, { 260.135, 31.8416, 26.4101 }, { 0.510576, -0.156086, -0.337325 }, { 0.0411204, 0.0490786, 0.0385724 } },
	{ "hematite", "hematite", { 0.00948374, 0.0117628, 0.00985037 }, { 0.0705694, 0.118965, 0.115059 }, { 0.000908552, 0.000601576, 0.00184248 }, { 0.515183, 0.498157, 0.73351 }, { 0.235045, 0.0609324, 0.00720315 }, { -0.120178, -0.096611, -0.235146 }, { 626.075, 967.655, 201.871 }, { 3.94735e-07, 3.54792e-07, 0.000101676 }, { 5.04376e-07, 16.4874, 2.78427e-07 }, { 24.7601, 14.7537, 31.4869 }, { -0.431506, 0.577853, -0.162174 }, { 0.0598105, 0.0605776, 0.0805566 } },
	{ "ipswich-pine-221", "ipswich-pine-221", { 0.0560746, 0.0222518, 0.0105117 }, { 0.0991995, 0.106719, 0.110343 }, { 0.014258, 0.0178759, 0.0188163 }, { 0.0625943, 0.113994, 0.118296 }, { 1.55288e-13, 6.595e-12, 3.97788e-13 }, { -0.0675784, -0.0696373, -0.0703103 }, { 68.7248, 53.3521, 50.6148 }, { 6.36482, 6.37738, 6.35839 }, { 111.962, 1.68378, 0.850892 }, { 20.956, 23.9475, 24.0837 }, { 0.842456, 0.66795, 0.641354 }, { 0.00364018, 0.00320748, 0.00300736 } },
	{ "light-brown-fabric", "", { 0.0612259, 0.0263619, 0.0187761 }, { 3.65487e-12, 9.7449e-12, 4.13685e-12 }, { 0.147778, 0.137639, 0.13071 }, { 0.188292, 0.188374, 0.189026 }, { 1, 1, 1 }, { 0, 0, 0 }, { 7.46192, 7.90085, 8.23842 }, { 3.59765, 3.74493, 3.85476 }, { 0.00660661, 0.0049581, 0.00395337 }, { 12.0657, 12.6487, 13.0977 }, { 0.221546, 0.213332, 0.206743 }, { 0.00126964, 0.000811029, 0.000670968 } },
	{ "light-red-paint", "paint-light-red", { 0.391162, 0.0458387, 0.0059411 }, { 0.522785, 0.144057, 0.214076 }, { 0.854048, 0.0627781, 0.205965 }, { 1.40232, 0.101554, 0.73691 }, { 0.0346142, 0.0655657, 0.0725632 }, { -0.0688995, 0.0400933, 0.0252314 }, { 4.68691, 16.2538, 4.71913 }, { 1.72476, 4.88106, 3.96894 }, { 0.0969212, 0.0124952, 1.02202e-07 }, { 5.57044, 16.1561, 19.9884 }, { 0.249637, 0.388724, -0.505514 }, { 0.00271456, 0.000889762, 0.000505975 } },
	{ "maroon-plastic", "maroon-plastic", { 0.189951, 0.0353828, 0.0321504 }, { 0.127693, 0.100703, 0.115731 }, { 0.00160715, 0.00110827, 0.00100127 }, { 0.684406, 0.645917, 0.569111 }, { 0.0479368, 0.0624437, 0.0921161 }, { -0.0134224, 0.00888653, 0.0423901 }, { 257.032, 398.91, 515.363 }, { 4.57088e-05, 2.62643e-05, 3.15876e-06 }, { 3.05979e-07, 2.76112e-07, 3.67984e-07 }, { 29.9497, 31.8569, 27.4899 }, { -0.211122, -0.159036, -0.309239 }, { 0.0285283, 0.0277773, 0.0286259 } },
	{ "natural-209", "natural-209", { 0.0961753, 0.0349012, 0.0121752 }, { 0.0781649, 0.0898869, 0.111321 }, { 0.0137282, 0.0154247, 0.0233645 }, { 0.0491415, 0.0673163, 0.149331 }, { 0.0522205, 0.0456428, 0.0279324 }, { -0.0277529, -0.0308045, -0.0492147 }, { 71.8965, 63.4418, 40.2168 }, { 6.3511, 6.32932, 6.32009 }, { 224.515, 48.774, 0.0171617 }, { 20.2337, 21.0421, 25.7335 }, { 0.874926, 0.811948, 0.483785 }, { 0.00465507, 0.00387214, 0.00295941 } },
	{ "neoprene-rubber", "neoprene-rubber", { 0.259523, 0.220477, 0.184871 }, { 0.275058, 0.391429, 0.0753145 }, { 0.143818, 0.207586, 0.0764912 }, { 0.770284, 0.774203, 0.700644 }, { 0.113041, 0.110436, 0.16895 }, { 0.060346, 0.0565499, 0.0788468 }, { 5.56845, 4.61088, 8.84784 }, { 3.02411, 3.82334, 2.38942 }, { 5.3042e-08, 1.01087e-07, 6.70643e-08 }, { 23.2109, 20.1103, 28.3099 }, { -0.379072, -0.500903, -0.156114 }, { 0.00218754, 0.00314666, 0.00180987 } },
	{ "nickel", "nickel", { 0.0144009, 0.0115339, 0.00989042 }, { 0.157696, 0.293022, 0.450103 }, { 0.00556292, 0.00627392, 0.00660563 }, { 0.171288, 0.168324, 0.161023 }, { 2.21884, 1.61986, 0.931645 }, { 1.21884, 1.22103, 0.698939 }, { 160.907, 143.234, 136.933 }, { 6.76252, 6.76447, 6.76278 }, { 16179.3, 3451.04, 3453.27 }, { 36.1378, 35.1378, 33.614 }, { 0.846825, 0.821209, 0.830948 }, { 0.132612, 0.177506, 0.155358 } },
	{ "nylon", "nylon", { 0.204199, 0.211192, 0.19234 }, { 0.156797, 0.303324, 0.236394 }, { 0.0250344, 0.0436802, 0.0421753 }, { 0.528875, 0.617086, 0.620808 }, { 0.240279, 0.117115, 0.127421 }, { 0.219191, 0.0940848, 0.101339 }, { 26.4669, 14.8484, 15.213 }, { 2.32486, 2.22618, 2.18663 }, { 1.17309e-07, 5.34378e-08, 6.68165e-08 }, { 39.9707, 33.9774, 34.3152 }, { 0.11793, -0.0190508, -0.00222626 }, { 0.00830775, 0.00885046, 0.00645556 } },
	{ "orange-paint", "orange-paint", { 0.368088, 0.147113, 0.00692426 }, { 0.524979, 0.116386, 0.199437 }, { 0.818115, 0.064743, 0.229391 }, { 1.44385, 0.0709512, 0.483597 }, { 6.92565e-13, 0.106161, 0.102279 }, { -0.174318, 0.0934385, 0.0625648 }, { 4.57466, 16.0185, 4.96427 }, { 1.84547, 4.70387, 3.6232 }, { 0.072629, 0.0299825, 0.000333551 }, { 5.96872, 14.9466, 13.2194 }, { 0.222125, 0.438216, -0.0759733 }, { 0.00178442, 0.000789226, 0.000301814 } },
	{ "pearl-paint", "pearl-paint", { 0.181967, 0.159068, 0.143348 }, { 0.105133, 0.0928717, 0.0802367 }, { 0.0724063, 0.0808503, 0.0596139 }, { 0.194454, 0.203296, 0.091958 }, { 0.168966, 0.297431, 0.401185 }, { -0.831034, -0.702569, -0.226546 }, { 13.6335, 12.3113, 17.1256 }, { 5.07525, 4.9097, 4.92468 }, { 0.000252814, 0.000258641, 0.0180128 }, { 18.9547, 18.2047, 16.1366 }, { 0.156731, 0.135808, 0.415562 }, { 0.00244402, 0.00167924, 0.0012928 } },
	{ "pickled-oak-260", "pickled-oak-260", { 0.181735, 0.14142, 0.125486 }, { 0.0283411, 0.0296418, 0.025815 }, { 0.0105853, 0.0102771, 0.0101188 }, { 2.31337e-14, 2.35272e-14, 1.99762e-14 }, { 5.38184e-13, 2.15933e-13, 3.55496e-12 }, { -0.309259, -0.291046, -0.329625 }, { 95.4759, 98.3089, 99.831 }, { 6.37433, 6.39032, 6.39857 }, { 4641.15, 5970.15, 6818.76 }, { 18.1707, 18.2121, 18.2334 }, { 1.00563, 1.01274, 1.01645 }, { 0.00336548, 0.0032891, 0.00330829 } },
	{ "pink-fabric2", "", { 0.24261, 0.0829238, 0.0751196 }, { 0.161823, 0.0591236, 0.00907967 }, { 0.220011, 0.148623, 0.111966 }, { 6.44517e-12, 3.24286e-13, 3.83556e-11 }, { 0.242032, 0.456181, 3.11925 }, { -0.0582931, 0.295844, 2.84268 }, { 5.66376, 7.80657, 9.98941 }, { 2.43742, 3.05804, 3.53392 }, { 0.111404, 0.079067, 0.0738938 }, { 7.47166, 9.23588, 10.5653 }, { 0.408014, 0.422779, 0.448866 }, { 0.00145188, 0.000903357, 0.00103814 } },
	{ "pink-fabric", "fabric-pink", { 0.270553, 0.223977, 0.240993 }, { 0.299998, 0.418074, 4.07112e-13 }, { 0.787023, 0.234345, 0.17346 }, { 1.77629, 4.64947e-15, 0.203846 }, { 0.121124, 0.0271041, 1 }, { 0.0151107, 0.00273969, 0 }, { 4.71724, 5.3941, 6.55218 }, { 2.23471, 2.34423, 3.31887 }, { 0.0266324, 0.11955, 0.00966123 }, { 7.20596, 7.20326, 11.0696 }, { 0.130136, 0.407562, 0.22282 }, { 0.00129262, 0.00124216, 0.00262093 } },
	{ "pink-felt", "fabric-pink-felt", { 0.259533, 0.192978, 0.185581 }, { 0.359813, 0.533498, 0.0390541 }, { 0.46679, 0.203504, 0.314933 }, { 0.663613, 1.70267e-13, 0.778919 }, { 0.0530603, 0.0108612, 1.12976 }, { -0.124124, -0.0455941, 1.05206 }, { 3.6266, 6.02293, 3.92458 }, { 2.22042, 2.55548, 3.68735 }, { 0.0333497, 0.102524, 8.10122e-05 }, { 7.40195, 7.81026, 13.5108 }, { 0.185696, 0.409228, -0.236166 }, { 0.000823299, 0.00122612, 0.00141316 } },
	{ "pink-jasper", "pink-jasper", { 0.226234, 0.138929, 0.110785 }, { 0.0846118, 0.0984038, 0.078693 }, { 0.00223592, 0.00203213, 0.0013737 }, { 0.53138, 0.4995, 0.436111 }, { 0.172825, 0.129714, 0.154397 }, { 0.0911362, 0.0599408, 0.0851167 }, { 253.003, 292.741, 474.2 }, { 1.64528, 1.86649, 5.29965e-08 }, { 5.71153e-08, 6.24765e-08, 15.5452 }, { 110.886, 113.953, 8.17872 }, { 0.416268, 0.422653, 0.561605 }, { 0.0166667, 0.0194625, 0.0142451 } },
	{ "pink-plastic", "pink-plastic", { 0.354572, 0.0905002, 0.0696372 }, { 0.0316585, 0.0444153, 6.36158e-15 }, { 0.0566727, 0.0369011, 0.142628 }, { 0.215149, 0.0621896, 0.189286 }, { 0.317781, 0.0925394, 1 }, { 0.266863, 0.0652821, 0 }, { 16.788, 27.3043, 7.67451 }, { 5.62766, 5.4919, 3.67439 }, { 1.71214e-05, 0.187553, 0.00561921 }, { 23.0822, 17.5369, 12.3737 }, { 0.0833775, 0.577298, 0.215985 }, { 0.00219041, 0.000905537, 0.00219415 } },
	{ "polyethylene", "polyethylene", { 0.228049, 0.239339, 0.240326 }, { 0.0420869, 0.134269, 0.0867928 }, { 0.0472725, 0.260465, 0.0719615 }, { 2.09548e-13, 0.743064, 2.27838e-13 }, { 0.489907, 0.316434, 0.181688 }, { 0.424297, 0.244434, 0.0772152 }, { 22.178, 4.24558, 14.9332 }, { 4.92064, 4.37149, 4.27422 }, { 0.330549, 9.78063e-07, 0.109488 }, { 14.3439, 17.178, 12.5991 }, { 0.61045, -0.470479, 0.517652 }, { 0.00285999, 0.00147486, 0.00191841 } },
	{ "polyurethane-foam", "polyurethane-foam", { 0.0898318, 0.0428583, 0.0340984 }, { 4.0852e-12, 8.0217e-14, 4.05682e-14 }, { 0.154984, 0.142104, 0.139418 }, { 0.188586, 0.188095, 0.188124 }, { 1, 1, 1 }, { 0, 0, 0 }, { 7.18433, 7.70043, 7.81983 }, { 3.50087, 3.67779, 3.7174 }, { 0.0079259, 0.00567317, 0.00525063 }, { 11.6922, 12.3795, 12.537 }, { 0.226848, 0.217315, 0.215132 }, { 0.00119032, 0.000823112, 0.000753708 } },
	{ "pure-rubber", "pure-rubber", { 0.284259, 0.251873, 0.223824 }, { 0.542899, 0.598765, 0.142162 }, { 0.62899, 0.41413, 0.0693873 }, { 1.13687, 0.452063, 0.262952 }, { 0.0185379, 0.00420901, 0.0353132 }, { -0.0265343, -0.035723, 0.0236736 }, { 3.75577, 3.75969, 13.6137 }, { 2.17187, 2.13584, 5.56595 }, { 0.0311422, 0.0526357, 7.8942e-07 }, { 7.14885, 7.0586, 23.9264 }, { 0.152489, 0.245615, -0.0979145 }, { 0.00177602, 0.00171837, 0.0007762 } },
	{ "purple-paint", "purple-paint", { 0.290743, 0.0347118, 0.0339802 }, { 0.301308, 0.205258, 0.280717 }, { 0.0339173, 0.00839425, 0.0378011 }, { 0.703279, 0.20083, 0.797376 }, { 0.0654958, 0.0786512, 0.0696424 }, { 0.0299494, 0.0467651, 0.0373649 }, { 16.3705, 104.698, 13.2101 }, { 1.72814, 6.77684, 1.54679 }, { 1.10548e-07, 0.177314, 6.66798e-08 }, { 38.0847, 41.1712, 38.2781 }, { 0.0789738, 0.576929, 0.0587144 }, { 0.00296394, 0.00277256, 0.00225685 } },
	{ "pvc", "pvc", { 0.0322978, 0.0357449, 0.0403426 }, { 0.28767, 0.317369, 0.310067 }, { 0.0171547, 0.0176681, 0.0213663 }, { 0.769726, 0.730637, 0.797555 }, { 0.0289552, 0.026258, 0.0281305 }, { 0.00815373, 0.00651989, 0.00714485 }, { 25.8231, 26.9026, 20.5927 }, { 1.20095, 1.31883, 1.23655 }, { 1.02768e-07, 1.09767e-07, 5.97648e-08 }, { 51.5975, 50.0184, 48.1953 }, { 0.218049, 0.208814, 0.173894 }, { 0.00471012, 0.00620207, 0.00575443 } },
	{ "red-fabric2", "", { 0.155216, 0.0226757, 0.0116884 }, { 1.80657e-15, 5.51946e-13, 1.35221e-15 }, { 0.16689, 0.135884, 0.128307 }, { 0.184631, 0.18856, 0.1883 }, { 1, 1, 1 }, { 0, 0, 0 }, { 6.78759, 7.98303, 8.36701 }, { 3.33819, 3.77225, 3.89063 }, { 0.0112096, 0.00468671, 0.00372538 }, { 11.0557, 12.7596, 13.2393 }, { 0.240935, 0.21164, 0.206003 }, { 0.00263226, 0.00060129, 0.000538747 } },
	{ "red-fabric", "fabric-red", { 0.201899, 0.0279008, 0.0103965 }, { 0.168669, 0.0486346, 0.040485 }, { 0.324447, 0.228455, 0.109436 }, { 0.787411, 0.821197, 0.279212 }, { 0.0718348, 0.0644687, 0.0206123 }, { -0.0585917, -0.0062547, -0.050402 }, { 3.88129, 4.32067, 9.16355 }, { 3.59825, 3.93046, 4.66379 }, { 0.000130047, 1.04152e-07, 4.59182e-05 }, { 13.0776, 19.649, 17.8852 }, { -0.209387, -0.530789, -0.0242035 }, { 0.00103676, 0.000248481, 0.000220698 } },
	{ "red-metallic-paint", "ch-ball-red-metallic", { 0.0380897, 0.00540095, 0.00281156 }, { 0.0416724, 0.07642, 0.108438 }, { 0.00133258, 0.00106883, 0.00128863 }, { 0.693854, 0.52857, 0.539477 }, { 2.45718e-16, 0.0598671, 0.0633332 }, { -1, -0.08904, -0.0114056 }, { 300.371, 521.418, 425.982 }, { 6.45857e-05, 6.3446e-07, 8.51754e-07 }, { 2.75773e-07, 4.05125e-07, 3.41703e-07 }, { 33.0213, 24.3646, 23.5793 }, { -0.121415, -0.469753, -0.532083 }, { 0.287672, 0.0467824, 0.0286558 } },
	{ "red-phenolic", "red-phenolic", { 0.165227, 0.0256259, 0.00935644 }, { 0.240561, 0.360634, 0.475777 }, { 0.0052844, 0.00467439, 0.00613717 }, { 0.568938, 0.509763, 0.575762 }, { 0.156419, 0.0972193, 0.069671 }, { 0.0752589, 0.0444558, 0.0266428 }, { 104.336, 128.839, 89.6357 }, { 1.57629, 1.92067, 1.57462 }, { 6.38793e-08, 6.71457e-08, 5.25172e-08 }, { 77.4088, 79.3795, 73.0158 }, { 0.343482, 0.352913, 0.324912 }, { 0.0680783, 0.0859239, 0.0936197 } },
	{ "red-plastic", "red-plastic", { 0.247569, 0.049382, 0.0175621 }, { 0.406976, 0.151478, 0.176348 }, { 0.28723, 0.0572489, 0.0624682 }, { 0.939617, 0.0851973, 0.0701483 }, { 0.10036, 0.178468, 0.149441 }, { 0.0512697, 0.13191, 0.102958 }, { 3.80564, 17.8374, 16.5631 }, { 4.37031, 4.96092, 4.75924 }, { 1.03822e-07, 0.0236845, 0.0321546 }, { 18.6088, 16.1371, 15.1243 }, { -0.609047, 0.43547, 0.445882 }, { 0.00134715, 0.000536577, 0.00047104 } },
	{ "red-specular-plastic", "red-plastic-specular", { 0.252589, 0.0397665, 0.0185317 }, { 0.0139957, 0.0343278, 0.0527973 }, { 6.01746e-05, 8.07327e-05, 0.000205705 }, { 0.174569, 0.202455, 0.390522 }, { 0.441328, 0.179378, 0.150221 }, { 0.191312, 0.0691835, 0.0760833 }, { 14623.1, 10620.2, 3332.98 }, { 22.6631, 12.9299, 3.06408 }, { 1e+38, 1e+38, 6.36761e-08 }, { 187.799, 162.245, 315.067 }, { 1.19812, 1.20136, 0.52102 }, { 0.0620224, 0.0541448, 0.0836154 } },
	{ "silicon-nitrade", "silicon-nitrade", { 0.0141611, 0.0115865, 0.00842477 }, { 0.0710113, 0.0670906, 0.015769 }, { 6.40406e-05, 0.000138867, 0.00224354 }, { 0.159422, 0.283527, 0.734323 }, { 0.0516164, 0.10318, 2.36643 }, { -0.0277792, -0.0531505, 1.36643 }, { 13926.7, 5668.82, 167.824 }, { 31.0579, 5.63563, 8.52534e-05 }, { 1e+38, 1e+38, 2.96855e-07 }, { 176.231, 148.745, 29.0282 }, { 1.20757, 1.14062, -0.241968 }, { 0.0712429, 0.0720965, 0.11404 } },
	{ "silver-metallic-paint2", "", { 0.0554792, 0.0573803, 0.0563376 }, { 0.121338, 0.115673, 0.10966 }, { 0.029859, 0.0303706, 0.0358666 }, { 0.144097, 0.104489, 0.158163 }, { 1.03749e-14, 3.52034e-15, 4.41778e-12 }, { -1, -1, -1 }, { 31.9005, 32.1514, 26.5685 }, { 6.08248, 5.89319, 5.95156 }, { 0.00761403, 0.0839948, 0.00138703 }, { 23.5891, 20.6786, 23.3297 }, { 0.435405, 0.54017, 0.34665 }, { 0.021384, 0.0187248, 0.0175762 } },
	{ "silver-metallic-paint", "metallic-silver", { 0.0189497, 0.0205686, 0.0228822 }, { 0.173533, 0.168901, 0.165266 }, { 0.037822, 0.038145, 0.0381908 }, { 0.165579, 0.162955, 0.160835 }, { 5.66903e-12, 1.65276e-14, 4.28399e-14 }, { -1, -1, -1 }, { 25.1551, 24.9957, 25.0003 }, { 5.92591, 5.90225, 5.89007 }, { 0.000684235, 0.000840795, 0.000995926 }, { 23.4725, 23.1898, 23.0144 }, { 0.310575, 0.317996, 0.324938 }, { 0.0122324, 0.0117672, 0.0113004 } },
	{ "silver-paint", "paint-silver", { 0.152796, 0.124616, 0.113375 }, { 0.30418, 0.30146, 0.283174 }, { 0.110819, 0.105318, 0.0785677 }, { 0.640378, 0.641115, 0.445228 }, { 2.37347e-13, 7.68194e-13, 2.9434e-12 }, { -0.350607, -0.355433, -0.359297 }, { 7.21531, 7.46519, 10.8289 }, { 3.06016, 2.97652, 3.78287 }, { 9.71666e-08, 1.09342e-07, 9.82336e-08 }, { 24.1475, 24.5083, 25.6757 }, { -0.281384, -0.257633, -0.200968 }, { 0.00361589, 0.00384995, 0.00444223 } },
	{ "special-walnut-224", "special-walnut-224", { 0.0121712, 0.00732998, 0.00463072 }, { 0.209603, 0.216118, 0.211885 }, { 0.117091, 0.119932, 0.131119 }, { 0.548899, 0.524858, 0.569425 }, { 0.0808859, 0.0802614, 0.0789982 }, { 0.0327605, 0.0324012, 0.0274637 }, { 7.42314, 7.41578, 6.77215 }, { 3.61532, 3.82778, 3.71096 }, { 1.19182e-07, 1.03098e-07, 1.08004e-07 }, { 22.9756, 22.7318, 22.2976 }, { -0.31158, -0.332257, -0.35462 }, { 0.000934753, 0.000878135, 0.000758769 } },
	{ "specular-black-phenolic", "black-bball", { 0.00212164, 0.00308282, 0.00410253 }, { 0.0881574, 0.0923246, 0.0398117 }, { 0.00119167, 0.000641898, 0.000186605 }, { 0.616914, 0.578026, 0.40121 }, { 0.122486, 0.0907984, 0.13855 }, { 0.0482851, 0.0348218, 0.0481813 }, { 395.656, 781.361, 3615.49 }, { 1.18051e-05, 1.22149, 2.87626 }, { 2.88932e-07, 6.09017e-08, 9.79363e-08 }, { 28.9748, 200.529, 331.336 }, { -0.257695, 0.487197, 0.524707 }, { 0.0311874, 0.0437836, 0.0853211 } },
	{ "specular-blue-phenolic", "blue-bball", { 0.00497564, 0.0138836, 0.032815 }, { 0.1077, 0.0898232, 0.175296 }, { 0.000918571, 0.0010348, 0.00176322 }, { 0.570978, 0.639916, 0.709385 }, { 0.0354139, 0.0488958, 0.023159 }, { -0.0434314, -0.0294427, -0.039938 }, { 558.482, 431.708, 222.547 }, { 3.57872e-06, 2.38985e-05, 6.87316e-05 }, { 3.33356e-07, 3.3336e-07, 3.2851e-07 }, { 28.5289, 32.1362, 30.3802 }, { -0.272244, -0.141224, -0.19005 }, { 0.0324614, 0.0338654, 0.057732 } },
	{ "specular-green-phenolic", "green-bball", { 0.00781782, 0.0259654, 0.0233739 }, { 0.0688449, 0.144658, 0.143654 }, { 0.000307494, 0.0010353, 0.00155331 }, { 0.365481, 0.585805, 0.787512 }, { 0.129429, 0.0443676, 5.423e-12 }, { 0.047932, -0.0136055, -0.0592743 }, { 2313.47, 482.847, 206.646 }, { 3.55591, 5.3306e-06, 0.000279403 }, { 1.03804e-07, 3.47154e-07, 3.16888e-07 }, { 256.625, 28.2901, 38.0691 }, { 0.511901, -0.277015, 0.00513058 }, { 0.0373526, 0.0450242, 0.0470867 } },
	{ "specular-maroon-phenolic", "maroon-bball", { 0.152486, 0.0263216, 0.00802748 }, { 0.0761775, 0.098375, 0.165913 }, { 0.000342958, 0.000605578, 0.00144136 }, { 0.4052, 0.553617, 0.65133 }, { 0.0646024, 0.0116325, 0.037551 }, { -0.0555983, -0.0527264, -0.0283844 }, { 1961.35, 868.119, 306.537 }, { 2.81751, 1.361, 2.43015e-05 }, { 5.70302e-08, 5.89096e-08, 3.27578e-07 }, { 248.591, 203.707, 29.053 }, { 0.506511, 0.488814, -0.242746 }, { 0.0465516, 0.0434438, 0.0545231 } },
	{ "specular-orange-phenolic", "orange-bball", { 0.32771, 0.0540131, 0.00883213 }, { 0.051915, 0.0686764, 0.0489478 }, { 7.91913e-05, 0.000139576, 1.62017e-05 }, { 0.253564, 0.354675, 3.55583e-12 }, { 0.0768695, 0.0496641, 0.0223538 }, { -0.015514, -0.023864, -0.0433847 }, { 10274.5, 5159.16, 61722.9 }, { 7.4207, 3.83488, 8.72207e+06 }, { 1e+38, 1.02522e-07, 2.30124e+07 }, { 170.364, 373.255, 855.811 }, { 1.19475, 0.53082, 0.608547 }, { 0.0405677, 0.0498947, 0.0953035 } },
	{ "specular-red-phenolic", "red-bball", { 0.303563, 0.0354891, 0.00899721 }, { 0.151819, 0.0938022, 0.196935 }, { 0.00117843, 0.00056476, 0.00185124 }, { 0.570146, 0.524406, 0.732785 }, { 0.0287445, 0.0672098, 0.0178433 }, { -0.0386172, 0.0144813, -0.0322308 }, { 438.968, 982.441, 201.328 }, { 2.88657e-06, 8.12418e-07, 9.95633e-05 }, { 3.8194e-07, 3.69661e-07, 3.03179e-07 }, { 26.0046, 30.0091, 31.3125 }, { -0.376008, -0.21799, -0.162894 }, { 0.0307055, 0.0199408, 0.0652829 } },
	{ "specular-violet-phenolic", "violet-bball", { 0.0686035, 0.0181856, 0.0210368 }, { 0.108459, 0.0471612, 0.171691 }, { 0.00123271, 0.000443974, 0.00149517 }, { 0.657484, 0.546753, 0.653065 }, { 0.0403569, 0.121081, 0.035323 }, { -0.0295013, 0.0563904, -0.0275623 }, { 351.208, 1193.45, 294.897 }, { 3.17585e-05, 1.3817, 2.44051e-05 }, { 3.02028e-07, 6.19706e-08, 3.40809e-07 }, { 31.3319, 234.879, 28.7237 }, { -0.168991, 0.500354, -0.252626 }, { 0.033584, 0.0495535, 0.0510203 } },
	{ "specular-white-phenolic", "white-bball", { 0.282896, 0.231703, 0.127818 }, { 0.0678467, 0.0683808, 0.032756 }, { 2.00918e-05, 3.22307e-05, 9.59333e-05 }, { 4.35028e-11, 0.131903, 0.390338 }, { 0.0132613, 0.0416359, 0.285115 }, { -0.1025, -0.0828341, 0.0759745 }, { 49772.5, 28318.4, 7131.79 }, { 4.58788e+06, 1.12234e+07, 3.05696 }, { 4.5813e+06, 1.00372e+07, 6.93295e-08 }, { 767.24, 646.93, 455.184 }, { 0.60999, 0.61958, 0.536789 }, { 0.134616, 0.114414, 0.179676 } },
	{ "specular-yellow-phenolic", "yellow-bball", { 0.309395, 0.135278, 0.0159106 }, { 0.0607659, 0.141526, 0.110839 }, { 0.000200174, 0.00179326, 0.00048094 }, { 0.355381, 0.767285, 0.505029 }, { 0.077039, 0.0175574, 0.038586 }, { -0.0182146, -0.0417745, -0.0114057 }, { 3597.35, 190.656, 1192.15 }, { 3.80702, 0.000181166, 1.69208 }, { 1.01181e-07, 3.0765e-07, 1.27637e-07 }, { 313.8, 34.1523, 221.08 }, { 0.523001, -0.0813947, 0.500075 }, { 0.0451868, 0.0491835, 0.031808 } },
	{ "ss440", "ss440", { 0.0229923, 0.0187037, 0.0153204 }, { 0.127809, 0.14899, 0.0473376 }, { 4.06782e-05, 4.86278e-05, 8.15367e-05 }, { 0.0888931, 0.259135, 0.332727 }, { 2.26479e-12, 5.7223e-13, 0.820577 }, { -0.474442, -0.223179, -0.179423 }, { 23197.8, 16625.7, 9083.17 }, { 1.43395e+07, 7.33446, 4.46921 }, { 1.06287e+07, 1e+38, 9.97795e-08 }, { 556.069, 215.021, 479.875 }, { 0.627834, 1.20059, 0.540075 }, { 0.366952, 0.463741, 0.874071 } },
	{ "steel", "steel", { 0.019973, 0.0127074, 0.0246402 }, { 0.0615275, 0.0469644, 0.0402151 }, { 8.63865e-05, 0.000249576, 5.77865e-05 }, { 0.3729, 0.583679, 0.471406 }, { 0.665193, 0.798139, 0.0115189 }, { -0.334807, -0.201861, -0.988481 }, { 8119.74, 1951.31, 10374.2 }, { 3.40501, 1.12086, 1.88549 }, { 1.25096e-07, 9.51373e-08, 1.20412e-07 }, { 474.668, 313.81, 603.334 }, { 0.539696, 0.519433, 0.545491 }, { 1.11996, 1.28262, 1.42004 } },
	{ "teflon", "teflon", { 0.276442, 0.263098, 0.260294 }, { 1.56924, 1.52804, 1.43859 }, { 0.678586, 0.662167, 0.577852 }, { 1.2402, 1.14126, 1.44077 }, { 9.40662e-13, 3.57656e-11, 1.50208e-11 }, { -0.0492032, -0.0587316, -0.0548028 }, { 3.91532, 3.82834, 3.64944 }, { 2.09871, 2.05044, 2.90315 }, { 0.0378525, 0.0435869, 0.0028469 }, { 6.8689, 6.72229, 9.65585 }, { 0.166634, 0.181605, -0.0468321 }, { 0.00462131, 0.00463132, 0.00408386 } },
	{ "tungsten-carbide", "tungsten-carbide", { 0.0151872, 0.0103016, 0.0123192 }, { 0.0504358, 0.075701, 0.0556673 }, { 6.6122e-05, 7.65809e-05, 4.80196e-05 }, { 0.255291, 0.270824, 0.26732 }, { 5.30357e-14, 5.44537e-12, 1.09586e-11 }, { -1, -1, -1 }, { 12280.7, 10423.3, 16683.5 }, { 7.42976, 6.36068, 6.81602 }, { 1e+38, 1e+38, 1e+38 }, { 185.623, 173.962, 218.975 }, { 1.197, 1.19586, 1.19745 }, { 0.472021, 0.583364, 0.850304 } },
	{ "two-layer-gold", "", { 0.0415046, 0.0312801, 0.0253658 }, { 1.58161, 1.18736, 1.63847 }, { 0.0263104, 0.0293804, 0.0241265 }, { 0.355682, 0.354281, 0.36415 }, { 0.117355, 0.0614942, 0.0447004 }, { 0.0411678, -0.0237579, -0.0100488 }, { 30.4478, 27.5367, 32.7278 }, { 3.91606, 3.99968, 3.79114 }, { 9.38513e-08, 6.62806e-08, 6.79906e-08 }, { 36.9091, 35.6578, 38.4915 }, { 0.0815181, 0.0465613, 0.0922721 }, { 0.165383, 0.160769, 0.168742 } },
	{ "two-layer-silver", "", { 0.0657916, 0.0595705, 0.0581288 }, { 1.55275, 2.00145, 1.93045 }, { 0.0149977, 0.0201665, 0.0225062 }, { 0.382631, 0.35975, 0.361657 }, { 4.93242e-13, 1.00098e-14, 0.0103259 }, { -0.0401315, -0.0395054, -0.0312454 }, { 50.1263, 38.8508, 34.9978 }, { 3.41873, 3.77545, 3.78138 }, { 6.09709e-08, 1.02036e-07, 1.01016e-07 }, { 46.6236, 40.8229, 39.1812 }, { 0.183797, 0.139103, 0.117092 }, { 0.170639, 0.189329, 0.21468 } },
	{ "violet-acrylic", "acrylic-violet", { 0.0599875, 0.023817, 0.0379025 }, { 0.134984, 0.13337, 0.295509 }, { 0.0011295, 0.00126481, 0.00186818 }, { 0.523244, 0.551606, 0.478149 }, { 0.100325, 0.0939057, 0.0663939 }, { 0.0603016, 0.0579001, 0.0353439 }, { 498.715, 424.307, 328.834 }, { 4.79917e-07, 1.39397e-06, 2.03376 }, { 4.27008e-07, 3.4655e-07, 1.05578e-07 }, { 23.5924, 24.3831, 116.7 }, { -0.515008, -0.476979, 0.432199 }, { 0.041711, 0.0411698, 0.0691096 } },
	{ "violet-rubber", "violet-rubber", { 0.223179, 0.0553634, 0.113238 }, { 0.547456, 0.0966027, 0.185463 }, { 0.445092, 0.089421, 0.1543 }, { 0.923481, 0.782897, 0.800907 }, { 0.0518927, 0.113508, 0.094549 }, { -0.0208955, 0.0585063, 0.036751 }, { 3.58673, 7.33539, 5.24404 }, { 2.83007, 2.27788, 3.00009 }, { 0.00374266, 9.65628e-08, 1.13746e-07 }, { 9.67534, 27.1898, 22.3519 }, { -0.0037936, -0.174481, -0.363226 }, { 0.00274248, 0.000994259, 0.00101746 } },
	{ "white-acrylic", "acrylic-white", { 0.314106, 0.300008, 0.263648 }, { 0.015339, 0.0736169, 0.0976209 }, { 0.00233914, 0.00168956, 0.00117717 }, { 0.570501, 0.51337, 0.440543 }, { 2.651, 0.38473, 0.249988 }, { 2.27139, 0.341493, 0.218093 }, { 226.134, 342.423, 548.396 }, { 1.41218, 2.17179e-07, 4.52325e-08 }, { 9.83895e-08, 5.84338e-06, 15.7173 }, { 110.209, 18.8595, 8.78916 }, { 0.419684, -0.61884, 0.56243 }, { 0.0537471, 0.0401864, 0.0400114 } },
	{ "white-diffuse-bball", "white-diffuse-bball", { 0.284726, 0.239199, 0.177227 }, { 0.680658, 0.828508, 0.502128 }, { 0.689731, 0.601478, 0.209084 }, { 1.23085, 0.956833, 0.800488 }, { 1.06465e-11, 7.45307e-19, 0.089058 }, { -0.109723, -0.090129, 0.0545089 }, { 3.94153, 3.65714, 4.53186 }, { 2.04845, 2.06125, 3.72838 }, { 0.0434345, 0.0439287, 1.06879e-07 }, { 6.69695, 6.79535, 20.1466 }, { 0.178983, 0.18898, -0.494774 }, { 0.007671, 0.0072205, 0.00441788 } },
	{ "white-fabric2", "", { 0.10784, 0.102669, 0.113943 }, { 0.0375359, 0.0296317, 0.0364218 }, { 0.0583526, 0.0520763, 0.063905 }, { 9.1836e-12, 5.41897e-13, 3.47708e-12 }, { 0.0528459, 0.0854122, 0.0860966 }, { 0.0578589, 0.0971296, 0.0985717 }, { 18.1669, 20.2291, 16.6809 }, { 4.60692, 4.77932, 4.46485 }, { 0.172556, 0.23927, 0.138435 }, { 13.5006, 13.9648, 13.1165 }, { 0.561147, 0.587176, 0.54153 }, { 0.00113864, 0.00114867, 0.00119185 } },
	{ "white-fabric", "", { 0.290107, 0.219835, 0.160654 }, { 0.230066, 0.156787, 2.19005e-12 }, { 0.479844, 0.196767, 0.162438 }, { 0.669662, 2.22894e-13, 0.194582 }, { 7.17773e-14, 0.0364632, 1 }, { -0.209237, -0.0528699, 0 }, { 3.60967, 6.18732, 6.91147 }, { 2.17095, 2.60737, 3.42418 }, { 0.0379118, 0.0990846, 0.00860197 }, { 7.221, 7.95861, 11.4289 }, { 0.195346, 0.409992, 0.225019 }, { 0.000728813, 0.000806025, 0.00177083 } },
	{ "white-marble", "", { 0.236183, 0.221746, 0.192889 }, { 0.24075, 0.221456, 0.23389 }, { 0.00430204, 0.00429388, 0.00405907 }, { 0.681306, 0.676061, 0.620701 }, { 0.118569, 0.167963, 0.140415 }, { 0.0620704, 0.102032, 0.071968 }, { 103.053, 104.312, 122.169 }, { 1.0396, 1.05866, 1.26372 }, { 1.08288e-07, 6.82995e-08, 5.76023e-08 }, { 88.7766, 89.0261, 88.95 }, { 0.378101, 0.372681, 0.371975 }, { 0.0493991, 0.0689784, 0.0584648 } },
	{ "white-paint", "white-paint", { 0.356024, 0.3536, 0.324889 }, { 4.14785, 0.255488, 0.108438 }, { 0.0654126, 0.0841905, 0.538225 }, { 0.796927, 0.7778, 0.888627 }, { 0.0770738, 0.25087, 1.75144e-13 }, { 0.0782297, 0.226285, -0.367441 }, { 8.89985, 7.66809, 3.59697 }, { 1.96428, 2.23069, 2.23051 }, { 5.9674e-08, 1.07673e-07, 0.0280623 }, { 31.0512, 27.6872, 7.41492 }, { -0.0896936, -0.151411, 0.154604 }, { 0.160631, 0.0657026, 0.0710504 } },
	{ "yellow-matte-plastic", "yellow-matte-plastic", { 0.276745, 0.108557, 0.0203686 }, { 0.806628, 1.99624, 0.977002 }, { 0.0237573, 0.0265756, 0.0305873 }, { 0.558101, 0.537647, 0.710679 }, { 0.0624565, 0.0605404, 0.0541077 }, { 0.049915, 0.0567466, 0.0473569 }, { 26.7398, 24.8437, 17.6137 }, { 2.1481, 2.32064, 1.65758 }, { 5.99553e-08, 5.96103e-08, 6.38097e-08 }, { 41.7723, 39.7008, 40.1979 }, { 0.11569, 0.0891281, 0.0886802 }, { 0.0469972, 0.10918, 0.0466045 } },
	{ "yellow-paint", "paint-yellow", { 0.288876, 0.195348, 0.0314583 }, { 0.449392, 0.412812, 0.168707 }, { 0.650734, 0.190849, 0.16131 }, { 1.21986, 0.0333524, 0.137577 }, { 0.00201415, 7.31639e-13, 0.13481 }, { -0.0897316, -0.0639292, 0.103378 }, { 3.83308, 6.31195, 7.06429 }, { 2.18358, 2.72018, 3.26561 }, { 0.0298708, 0.0764778, 0.0198928 }, { 7.1694, 8.39507, 10.5682 }, { 0.146537, 0.387991, 0.292185 }, { 0.00152911, 0.00123385, 0.000424783 } },
	{ "yellow-phenolic", "phenolic-yellow", { 0.26924, 0.190177, 0.0858303 }, { 0.0861694, 0.0960246, 0.122709 }, { 0.00126171, 0.00197611, 0.00187166 }, { 0.444222, 0.512534, 0.504223 }, { 0.0988294, 0.120249, 0.0929719 }, { 0.0295258, 0.0539044, 0.0471952 }, { 509.34, 294.418, 314.704 }, { 4.59604e-08, 1.76251, 1.81658 }, { 15.6472, 6.5671e-08, 9.96466e-08 }, { 8.52565, 116.04, 118.064 }, { 0.561753, 0.425536, 0.432459 }, { 0.0238647, 0.0190162, 0.0173945 } },
	{ "yellow-plastic", "yellow-plastic", { 0.221083, 0.193042, 0.0403393 }, { 0.265199, 0.340361, 0.0670333 }, { 0.280789, 0.146396, 0.0248514 }, { 0.920018, 0.0356237, 7.80502e-12 }, { 0.144471, 0.10845, 0.125678 }, { 0.113592, 0.0663735, 0.103644 }, { 3.85808, 7.84753, 41.2517 }, { 4.34432, 3.16869, 5.71439 }, { 1.04213e-07, 0.055929, 7.84273 }, { 18.6813, 9.73614, 16.4495 }, { -0.60265, 0.393386, 0.781122 }, { 0.00125483, 0.00140731, 0.00101943 } }
};

static double 
sgd__g1(const dir& k, double theta0, double c, double k_, double lambda)
{
	double tmp1 = max(0.0, k.theta - theta0);
	double tmp2 = 1.0 - exp(c * pow(tmp1, k_));
	double tmp3 = 1.0 + lambda * tmp2;
	return min(1.0, max(0.0, tmp3));
}

static double sgd__ndf(double cos_theta_h, double alpha, double p, double kap)
{
	const double inv_pi = 1.0 / M_PI;
	double c2 = cos_theta_h * cos_theta_h;
	double t2 = (1.0 - c2) / c2;
	double ax = alpha + t2 / alpha;

	return ((kap * exp(-ax) * inv_pi) / (pow(ax, p) * c2 * c2));
}

// -------------------------------------------------------------------------------------------------
// SGD Ctor
sgd::sgd(const char *name): m_fresnel(NULL), m_data(NULL)
{
	bool found = false;
	for (int i = 0; i < 100; ++i) {
		if (!strcmp(s_data[i].name, name)
		    || !strcmp(s_data[i].otherName, name)) {
			m_data = &s_data[i];
			m_fresnel = new fresnel::sgd(vec3::from_raw(m_data->f0),
			                             vec3::from_raw(m_data->f1));
			found = true;
			break;
		}
	}
	if (!found) throw exc("djb_error: No SGD parameters for %s\n", name);
}

// -------------------------------------------------------------------------------------------------
// SGD eval
vec3 sgd::eval(const dir& i, const dir& o) const {
	if (i.theta < M_PI * 0.5) {
		dir h, d;
		std_coords_to_half_diff_coords(i, o, h, d);

		const vec3 Ks = vec3::from_raw(m_data->rhoS);
		const vec3 Kd = vec3::from_raw(m_data->rhoD);
		vec3 F = eval_fresnel(d.theta);
		vec3 G = gaf(i, o, h);
		vec3 D = ndf(h);

		return ((Kd + Ks * (F * D * G) / (cos(i.theta) * cos(o.theta))) / M_PI);
	}
	return vec3(0);
}

// -------------------------------------------------------------------------------------------------
// SGD Shadowing
vec3 sgd::gaf(const dir& i, const dir& o, const dir& h) const
{
	return g1(i) * g1(o);
}

vec3 sgd::g1(const dir& k) const
{
	double g1[3];

	for (int i = 0; i < 3; ++i)
		g1[i] = sgd__g1(k, m_data->theta0[i], m_data->c[i],
	                    m_data->k[i], m_data->lambda[i]);

	return vec3::from_raw(g1);
}

// -------------------------------------------------------------------------------------------------
// SGD NDF
vec3 sgd::ndf(const dir& h) const
{
	double cos_theta_h = cos(h.theta);
	double ndf[3];

	for (int i = 0; i < 3; ++i)
		ndf[i] = sgd__ndf(cos_theta_h, m_data->alpha[i],
	                      m_data->p[i], m_data->kap[i]);

	return vec3::from_raw(ndf);
}


// *************************************************************************************************
// ABC Distribution API implementation

const abc::data abc::s_data[8] = {
	{ "alumina-oxide", "alumina-oxide", { 0.3109, 0.2887, 0.2534 }, { 2191.683921, 1851.525468, 2064.303439 }, 97381.554876, 1.393850, 1.437706},
	{ "blue-acrylic", "acrylic-blue", { 0.0111, 0.0351, 0.1043 }, { 1191.397279, 1099.257252, 1082.125405 }, 255868.959838, 1.084736, 1.475061},
	{ "blue-metallic-paint", "metallic-blue", { 0.0038, 0.0010, 0.0092 }, { 0.495104, 0.469843, 1.097868 }, 48.572060, 1.702520, 9.024582},
	{ "color-changing-paint1", "", { 0.0022, 0.0026, 0.0053 }, { 32.486814, 34.621836, 36.775240 }, 2456.161812, 1.388325, 2.066753},
	{ "gold-metallic-paint", "metallic-gold", { 0.0092, 0.0056, 0.0004 }, { 1.685630, 1.147257, 0.418757 }, 53.738525, 1.511730, 9.031934},
	{ "pink-plastic", "pink-plastic", { 0.3602, 0.0791, 0.0531 }, { 0.826036, 0.561110, 0.579990 }, 16122.769509, 0.217444, 1.810468},
	{ "purple-paint", "purple-paint", { 0.2802, 0.0312, 0.0323 }, { 14.006363, 13.769886, 13.753972 }, 408.684145, 1.776584, 1.483770},
	{ "two-layer-gold", "", { 0.0, 0.0, 0.0 }, { 103.922809, 91.724262, 76.325031 }, 481560.753343, 0.631262, 8.633638 }
};

static double abc__ndf(double cos_theta_h, double A, double B, double C)
{
	double tmp = 1.0 - cos_theta_h;

	return (A / pow(1.0 + B * tmp, C));
}

// -------------------------------------------------------------------------------------------------
// ABC Ctor
abc::abc(const char *name): m_fresnel(NULL), m_data(NULL)
{
	bool found = false;
	for (int i = 0; i < 100; ++i) {
		if (!strcmp(s_data[i].name, name)
		    || !strcmp(s_data[i].otherName, name)) {
			m_data = &s_data[i];
			m_fresnel = new fresnel::unpolarized(djb::vec3(m_data->ior)),
			found = true;
			break;
		}
	}
	if (!found) throw exc("djb_error: No ABC parameters for %s\n", name);
}

// -------------------------------------------------------------------------------------------------
// ABC eval
vec3 abc::eval(const dir& i, const dir& o) const {
	if (i.theta < M_PI * 0.5) {
		dir h, d;
		std_coords_to_half_diff_coords(i, o, h, d);

		const vec3 Kd = vec3::from_raw(m_data->kD);
		vec3 F = eval_fresnel(d.theta);
		double G = gaf(i, o, h);
		vec3 D = ndf(h);

		return (Kd / M_PI  + (F * D * G) / (M_PI * cos(i.theta) * cos(o.theta)));
	}
	return vec3(0);
}

// -------------------------------------------------------------------------------------------------
// ABC Shadowing
double abc::gaf(const dir& i, const dir& o, const dir& h) const
{
	if (h.theta < 0.5 * M_PI && o.theta < 0.5 * M_PI) {
		double cos_theta_h = cos(h.theta);
		double sin_theta_h = sin(h.theta);
		double cos_theta_o = cos(o.theta);
		double cos_theta_d = cos_theta_h * cos_theta_o
		                   + cos(h.phi - o.phi) * sin_theta_h 
		                   * sin(o.theta);
		double tmp = 2.0 * cos_theta_o * cos_theta_h / cos_theta_d;
		double masking = min(1.0, tmp);

		if (/*m_shadow*/true) {
			if (i.theta < 0.5 * M_PI) {
				double cos_theta_i = cos(i.theta);
				double cos_theta_d = cos_theta_h * cos_theta_i
				                   + cos(h.phi - i.phi) * sin_theta_h
				                   * sin(i.theta);

				double tmp = 2.0 * cos_theta_i * cos_theta_h / cos_theta_d;
				double masking_and_shadowing = min(masking, tmp);

				return masking_and_shadowing;
			}

			return 0.0; // below horizon
		}

		return masking;
	}

	return 0.0; // below horizon
}

// -------------------------------------------------------------------------------------------------
// ABC NDF
vec3 abc::ndf(const dir& h) const
{
	double cos_theta_h = cos(h.theta);
	double ndf[3];

	for (int i = 0; i < 3; ++i)
		ndf[i] = abc__ndf(cos_theta_h, m_data->A[i], m_data->B, m_data->C);

	return vec3::from_raw(ndf);
}


} // namespace djb

#endif // DJ_BRDF_IMPLEMENTATION

