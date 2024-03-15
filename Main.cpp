#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "../HeadFile/svpng.inc"

#include <omp.h>
#include <vector>
#include <random>
#include <numbers>
#include <iostream>

//color
const glm::vec3 RED(1, 0.0, 0.0);
const glm::vec3 GREEN(0.0, 1, 0.0);
const glm::vec3 BLUE(0.0, 0.0, 1);
const glm::vec3 YELLOW(1.0, 1.0, 0.1);
const glm::vec3 CYAN(0.1, 1.0, 1.0);
const glm::vec3 MAGENTA(1.0, 0.1, 1.0);
const glm::vec3 GRAY(0.5, 0.5, 0.5);
const glm::vec3 WHITE(1, 1, 1);

//生成图片的长宽
const int HEIGHT = 512, WIDTH = 512;
const float SCREEN_Z = 1.1;
const glm::vec3 CAMERA(0.0f, 0.0f, 4.0f);

const double PI = std::numbers::pi;
//半球采样的pdf
const float pdf_hemi = 1.0f / (2.0f * PI);
//每个像素的采样次数
const int SSP = 512;
//俄罗斯轮盘赌概率
const double RR = 0.8f;
//最大递归深度
const int MAX_DEPTH = 100;


//Random
inline double random_double()
{
	static std::uniform_real_distribution<double> distribution(0.0, 1.0);
	static std::mt19937 generator;
	return distribution(generator);
}

inline double random_double(double min, double max)
{
	return (max - min) * random_double() + min;
}
//剔除法生成在单位球内的单位向量
inline glm::vec3 random_in_unit_sphere()
{
	while (true)
	{
		glm::vec3 p(random_double(-1, 1), random_double(-1, 1), random_double(-1, 1));
		if (glm::dot(p, p) < 1) return glm::normalize(p);
	}
}


//生成一个在法向半球内的单位向量
inline glm::vec3 random_in_hemisphere(const glm::vec3& normal)
{
	glm::vec3 p = random_in_unit_sphere();
	if (glm::dot(p, normal) > 0.0) return p;
	else return -p;
}

//漫反射的一种实现：Lambertian反射
inline glm::vec3 Lambertian_vector(glm::vec3 normal)
{
	return glm::normalize(normal + random_in_unit_sphere());
}


struct Ray
{
	glm::vec3 origin = glm::vec3(0.0f);
	glm::vec3 direction = glm::vec3(0.0f);
};

class Material
{
public:

	Material() {}

	bool isEmissive = false;
	glm::vec3 color = glm::vec3(0.0f);
	//简言之，给定一条入射光线，计算该入射光线在交点的这种材质下的反射光线
	virtual Ray scattered(glm::vec3 hitPoint, glm::vec3 normal, glm::vec3& blendingColor, glm::vec3 in_direct) const = 0;
};
//漫反射材质
class Lambertian : public Material
{
public:
	Lambertian(glm::vec3 Color)
	{
		this->color = Color;
	}

	Ray scattered(glm::vec3 hitPoint, glm::vec3 normal, glm::vec3& blendingColor, glm::vec3 in_direct) const override
	{
		glm::vec3 sampling_dir(Lambertian_vector(normal));

		blendingColor = this->color;

		return Ray(hitPoint, sampling_dir);
	}
};
//反射材质
class Reflect : public Material
{
public:
	float reflectRate = 0.0f;
	float roughtness = 1.0f;

	Reflect(glm::vec3 Color)
	{
		this->color = Color;
	}

	Ray scattered(glm::vec3 hitPoint, glm::vec3 normal, glm::vec3& blendingColor, glm::vec3 in_direct) const override
	{

		double R(random_double());
		glm::vec3 sampling_dir;
		if (R < this->reflectRate)//镜面反射
		{
			glm::vec3 random_dir(random_in_hemisphere(normal));
			glm::vec3 reflect_dir(glm::reflect(in_direct, normal));
			sampling_dir = glm::normalize(glm::mix(reflect_dir, random_dir, this->roughtness));
		}
		else//漫反射
		{
			sampling_dir = Lambertian_vector(normal);
			blendingColor = this->color;
		}

		return Ray(hitPoint, sampling_dir);
	}

};
//折射材质
class Refract : public Material
{
public:

	Refract(float IR) : ir(IR) {}


	float ir;//该材质的折射率
	float roughtness = 1.0f; //粗糙程度

	Ray scattered(glm::vec3 hitPoint, glm::vec3 normal, glm::vec3& blendingColor, glm::vec3 in_direct) const override
	{
		float refraction_ratio = glm::dot(normal, in_direct) < 0 ? 1.0f / ir : ir;//通过判断光线是从空气射入材质还是从材质射出空气调整折射率比值。
		glm::vec3 unit_direct(glm::normalize(in_direct));

		float cos(glm::dot(-unit_direct, normal));
		float sin(sqrt(1.0f - cos * cos));
		glm::vec3 refracted_dir;


		if (refraction_ratio * sin > 1.0f || reflectance(cos, refraction_ratio) > random_double())
		{
			//只能进行反射
			refracted_dir = glm::reflect(unit_direct, normal);
			
		}
		else
		{
			//否则进行折射
			glm::vec3 sampling_dir = glm::refract(unit_direct, normal, refraction_ratio);
			refracted_dir = glm::mix(sampling_dir, -Lambertian_vector(normal), this->roughtness);
		}

		

		return Ray(hitPoint, refracted_dir);
	}

	static float reflectance(float cos, float refraction_ratio)
	{
		auto r0((1 - refraction_ratio) / (1 + refraction_ratio));
		r0 *= r0;
		return r0 + (1 - r0) * pow(1 - cos, 5);
	}
};


//光线与物体交点
struct HitResult
{
	bool isHit = false;
	float distance = 0.0f;
	glm::vec3 hitPoint = glm::vec3(0.0f);
	glm::vec3 normal = glm::vec3(0.0f);
	std::shared_ptr<Material> material;
};
//形状类
class Shape
{
public:
	Shape() {}
	virtual HitResult intersect(Ray ray) const = 0;
};
//三角形类
class Triangle : public Shape
{
public:
	Triangle() {}
	Triangle(glm::vec3 P0, glm::vec3 P1, glm::vec3 P2, std::shared_ptr<Material> material)
		: p0(P0), p1(P1), p2(P2)
	{
		this->material = material;
		this->normal = glm::normalize(glm::cross(p1 - p0, p2 - p0));
		this->center = (p0 + p1 + p2) / 3.0f;
	}
	//计算光线是否与该物体相交，并返回交点
	HitResult intersect(Ray ray) const override
	{
		HitResult res;

		glm::vec3 Normal = glm::dot(ray.direction, this->normal) > 0 ? -this->normal : this->normal;
		// 如果视线和三角形平行
		if (fabs(glm::dot(Normal, ray.direction)) < 0.00001f) return res;

		glm::vec3 E1 = p1 - p0, E2 = p2 - p0, S = ray.origin - p0;

		// 利用光线方向和E2的叉乘结果S1，以及S和E1的叉乘结果S2
		glm::vec3 S1 = glm::cross(ray.direction, E2), S2 = glm::cross(S, E1);

		float divisor = glm::dot(S1, E1);
		// 当除数接近于0时，光线与三角形平行，无交点
		if (fabs(divisor) < 0.00001f) return res;
		float invDivisor = 1.0f / divisor;

		float b1 = glm::dot(S1, S) * invDivisor;
		float b2 = glm::dot(S2, ray.direction) * invDivisor;
		float t = glm::dot(S2, E2) * invDivisor;

		// 检查贝塞尔坐标是否在三角形内部，并确保t大于一个阈值以避免自交
		if (b1 < 0.0f || b1 > 1.0f || b2 < 0.0f || (b1 + b2) > 1.0f || t < 0.005f) return res;

		res.isHit = true;
		res.distance = t;
		res.hitPoint = ray.origin + t * ray.direction;
		res.material = material;
		res.normal = Normal;
		return res;
	}

	glm::vec3 p0, p1, p2;
	glm::vec3 center;
	glm::vec3 normal;
	std::shared_ptr<Material> material;
};
//球类
class Sphere : public Shape {
public:
	Sphere() {}
	Sphere(glm::vec3 Center, float Radius, std::shared_ptr<Material> material)
		: center(Center), radius(Radius)
	{
		this->material = material;
	}
	//计算光线是否与该物体相交，并返回交点
	HitResult intersect(Ray ray) const override
	{
		HitResult res;
		glm::vec3 L = center - ray.origin;
		float t_center = glm::dot(L, ray.direction);
		if (t_center < 0) return res;

		float d2(glm::dot(L, L) - t_center * t_center);
		float r2(radius * radius);

		if (d2 > r2) return res;

		float t_hc(sqrt(r2 - d2));
		float t0(t_center - t_hc);
		float t1(t_center + t_hc);

		if (t0 > t1) std::swap(t0, t1);

		const float epsilon = 0.0001f;
		if (t0 < epsilon)
		{
			t0 = t1;
			if (t1 < epsilon) return res;
		}

		res.isHit = true;
		res.distance = t0;
		res.hitPoint = ray.origin + t0 * ray.direction;
		res.material = material;
		glm::vec3 normal = glm::normalize(res.hitPoint - center);
		res.normal = glm::dot(ray.direction, normal) > 0 ? -normal : normal; // 根据射线方向调整法线方向

		return res;
	}

	glm::vec3 center;
	float radius;
	std::shared_ptr<Material> material;
};

//计算一根光线与全部物体的交点，并返回最近的交点
HitResult intersect(std::vector<Shape*>& shapes, Ray ray)
{
	HitResult res, r;
	res.distance = std::numeric_limits<float>::infinity();

	for (auto& shape : shapes)
	{
		r = shape->intersect(ray);
		if (r.isHit && r.distance < res.distance) res = r;
	}
	return res;
}
//路径追踪函数，返回该光线的颜色
glm::vec3 PathTracing(std::vector<Shape*>& shapes, Ray ray, int depth)
{

	if (depth >= MAX_DEPTH) return glm::vec3(0.0f);
	HitResult res = intersect(shapes, ray);

	if (res.isHit)
	{
		//如果交点是光源，则直接返回光源颜色
		if (res.material->isEmissive) return res.material->color;
		//如果交点不是光源，则随机确定一个方向采样
		//俄罗斯轮盘赌让光线以一定概率停下
		double P = random_double();
		if (P > RR) return glm::vec3(0.0f);

		double R(random_double());
		glm::vec3 blendingColor(1.0f);

		Ray sampling_ray = res.material->scattered(res.hitPoint, res.normal, blendingColor, ray.direction);


		HitResult sampling_res = intersect(shapes, sampling_ray);

		float cosine(glm::dot(res.normal, sampling_ray.direction));


		if (sampling_res.isHit)
		{
			return PathTracing(shapes, sampling_ray, depth + 1) * blendingColor * cosine / (float)RR;
		}
	}

	return glm::vec3(0.0f);
}

int main()
{
	std::vector<Shape*> shapes;

	auto white_reflect1(std::make_shared<Reflect>(WHITE));
	white_reflect1->reflectRate = 0.8f;
	white_reflect1->roughtness = 0.1f;

	auto refracted(std::make_shared<Refract>(1.5f));
	refracted->roughtness = 0.0f;

	Sphere s1(glm::vec3(0.6, -0.7, -0.5), 0.3, std::make_shared<Lambertian>(WHITE));
	Sphere s2(glm::vec3(0.0, -0.2, -0.5), 0.3, refracted);
	Sphere s3(glm::vec3(-0.6, -0.7, -0.5), 0.3, white_reflect1);


	shapes.push_back(&s1);
	shapes.push_back(&s2);
	shapes.push_back(&s3);
	//在场景中央添加一个黄色三角形
	shapes.push_back(new Triangle(glm::vec3(-0.2, -0.2, -0.95), glm::vec3(0.2, -0.2, -0.95), glm::vec3(-0.0, -0.9, -0.95), std::make_shared<Lambertian>(YELLOW)));

	// 发光物
	Triangle l1 = Triangle(glm::vec3(0.4, 0.99, 0.4), glm::vec3(-0.4, 0.99, -0.4), glm::vec3(-0.4, 0.99, 0.4), std::make_shared<Lambertian>(WHITE));
	Triangle l2 = Triangle(glm::vec3(0.4, 0.99, 0.4), glm::vec3(0.4, 0.99, -0.4), glm::vec3(-0.4, 0.99, -0.4), std::make_shared<Lambertian>(WHITE));
	l1.material->isEmissive = true;
	l2.material->isEmissive = true;
	shapes.push_back(&l1);
	shapes.push_back(&l2);

	// 背景盒子
	// bottom
	shapes.push_back(new Triangle(glm::vec3(1, -1, 1), glm::vec3(-1, -1, -1), glm::vec3(-1, -1, 1), std::make_shared<Lambertian>(WHITE)));
	shapes.push_back(new Triangle(glm::vec3(1, -1, 1), glm::vec3(1, -1, -1), glm::vec3(-1, -1, -1), std::make_shared<Lambertian>(WHITE)));
	// top
	shapes.push_back(new Triangle(glm::vec3(1, 1, 1), glm::vec3(-1, 1, 1), glm::vec3(-1, 1, -1), std::make_shared<Lambertian>(WHITE)));
	shapes.push_back(new Triangle(glm::vec3(1, 1, 1), glm::vec3(-1, 1, -1), glm::vec3(1, 1, -1), std::make_shared<Lambertian>(WHITE)));
	// back
	shapes.push_back(new Triangle(glm::vec3(1, -1, -1), glm::vec3(-1, 1, -1), glm::vec3(-1, -1, -1), std::make_shared<Lambertian>(CYAN)));
	shapes.push_back(new Triangle(glm::vec3(1, -1, -1), glm::vec3(1, 1, -1), glm::vec3(-1, 1, -1), std::make_shared<Lambertian>(CYAN)));
	// left
	shapes.push_back(new Triangle(glm::vec3(-1, -1, -1), glm::vec3(-1, 1, 1), glm::vec3(-1, -1, 1), std::make_shared<Lambertian>(BLUE)));
	shapes.push_back(new Triangle(glm::vec3(-1, -1, -1), glm::vec3(-1, 1, -1), glm::vec3(-1, 1, 1), std::make_shared<Lambertian>(BLUE)));
	// right
	shapes.push_back(new Triangle(glm::vec3(1, 1, 1), glm::vec3(1, -1, -1), glm::vec3(1, -1, 1), std::make_shared<Lambertian>(RED)));
	shapes.push_back(new Triangle(glm::vec3(1, -1, -1), glm::vec3(1, 1, 1), glm::vec3(1, 1, -1), std::make_shared<Lambertian>(RED)));
	double* image = new double[HEIGHT * WIDTH * 3];
	memset(image, 0.0, sizeof(double) * WIDTH * HEIGHT * 3);
	omp_set_num_threads(50); // 设置期望的线程数量

#pragma omp parallel for collapse(2) // 并行化外部和内部循环
	for (int i = 0; i < HEIGHT; i++)
	{
		for (int j = 0; j < WIDTH; j++)
		{
			//从世界坐标到图像坐标的转化
			double x = 2.0 * j / double(WIDTH) - 1.0;
			double y = 2.0 * (HEIGHT - i) / double(HEIGHT) - 1.0;

			// MSAA
			x += (random_double(-0.5f, 0.5f)) / double(WIDTH);
			y += (random_double(-0.5f, 0.5f)) / double(HEIGHT);

			//计算光线方向
			glm::vec3 pixel_poi(x, y, SCREEN_Z);
			glm::vec3 ray_direction = glm::normalize(pixel_poi - CAMERA);
			Ray ray(CAMERA, ray_direction);

			glm::vec3 color(0.0f);
			for (int k = 0; k < SSP; k++) // SSP 是每个像素的采样数
			{
				//多次采样，投射光线计算颜色值，
				color += PathTracing(shapes, ray, 0) / pdf_hemi;
			}

			color /= static_cast<float>(SSP);

			// 计算索引位置
			int idx = (i * WIDTH + j) * 3;
			image[idx + 0] = color.x;
			image[idx + 1] = color.y;
			image[idx + 2] = color.z;
		}
	}

	unsigned char* c_image = new unsigned char[WIDTH * HEIGHT * 3];// 图像buffer
	unsigned char* c_p = c_image;
	double* S = image;
	for (int i = 0; i < HEIGHT; i++)
	{
		for (int j = 0; j < WIDTH; j++)
		{
			*c_p++ = (unsigned char)glm::clamp(pow(*S++, 1.0f / 2.2f) * 255, 0.0, 255.0);  // R 通道
			*c_p++ = (unsigned char)glm::clamp(pow(*S++, 1.0f / 2.2f) * 255, 0.0, 255.0);  // G 通道
			*c_p++ = (unsigned char)glm::clamp(pow(*S++, 1.0f / 2.2f) * 255, 0.0, 255.0);  // B 通道
		}
	}
	FILE* fp;
	fopen_s(&fp, "image.png", "wb");

	svpng(fp, WIDTH, HEIGHT, c_image, 0);
	fclose(fp);


	return 0;
}