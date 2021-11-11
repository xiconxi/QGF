// Copyright 2011-2019 the Polygon Mesh Processing Library developers.
// Distributed under a MIT-style license, see LICENSE.txt for details.

#include <pmp/visualization/MeshViewer.h>
#include <pmp/algorithms/SurfaceCurvature.h>
#include <pmp/algorithms/SurfaceSmoothing.h>
#include <imgui.h>
#include <pmp/algorithms/Quadric.h>
#include <pmp/algorithms/DifferentialGeometry.h>
#include <pmp/algorithms/SurfaceNormals.h>
#include <set>
#include <omp.h>


#define OMP_PARALLEL _Pragma("omp parallel")
#define OMP_FOR _Pragma("omp for")
#define OMP_SINGLE _Pragma("omp single")

namespace pmp {

class QGFSmoothing {
public:
    QGFSmoothing(SurfaceMesh& mesh): mesh_(mesh) {
        pmp::SurfaceNormals::compute_face_normals(mesh_);
        f_quadric_ = mesh_.face_property<Quadric>("f:quadric", Quadric());
        v_quadric_ = mesh_.vertex_property<Quadric>("v:quadric", Quadric());
        f_normal_ = mesh_.face_property<Normal>("f:normal", Normal());
    }

    double mean_edge_length() const {
        double len_sum = 0;
        for(auto e: mesh_.edges())
            len_sum += mesh_.edge_length(e);
        return len_sum / mesh_.n_edges();
    }

    // destructor
    ~QGFSmoothing() {}

    void do_smoothing(unsigned int iters, double s_b, double s_s, double s_r) {
        compute_face_quadric();
        for(int i = 0; i < iters; i++) {
            quadric_diffusion(5, s_s, s_r);
            compute_vertex_quadric(s_b);
            quadric_motion(10);
        }
    }

private:

    void quadric_motion(int n_iter) {
        auto vnormal = mesh_.vertex_property<Normal>("v:normal");
        for(int i_iter = 0; i_iter < n_iter; i_iter++){
            std::cout <<"\r motion " << i_iter << std::endl;
            SurfaceNormals::compute_vertex_normals(mesh_);
            double max_lambda = 0;
            OMP_PARALLEL
            {
                OMP_FOR
                for (int i = 0; i < mesh_.n_vertices(); i++)
                {
                    auto v = pmp::Vertex(i);
                    double lambda = v_quadric_[v].vertex_gradient(
                        vnormal[v], mesh_.position(v));
                    max_lambda = std::max(max_lambda, lambda);
                    if (norm(vnormal[v]) == NAN || lambda == NAN)
                        continue;
                    mesh_.position(v) += vnormal[v] * lambda;
                }
            }
        }
    }

    double quadric_distance(const Quadric& Q, Face f) {
        double quadric = 0;
        auto vit = mesh_.vertices(f);
        quadric += ( Q( mesh_.position(*vit) ) );
        quadric += ( Q( mesh_.position(*++vit) ) );
        quadric += ( Q( mesh_.position(*++vit) ) );
        return quadric* triangle_area(mesh_, f)/3.0f;
    }

    void quadric_diffusion(int n_iter, double sigma_s = 0.1, double sigma_r = 0.1) {
        FaceProperty<Quadric> new_Qf = mesh_.face_property<Quadric>("f:quadric_");
        for(int i_iter = 0; i_iter < n_iter; i_iter++) {
            std::cout <<"\r diffusion " << i_iter <<std::endl;
            OMP_PARALLEL { OMP_FOR
            for(int i = 0; i < mesh_.n_faces(); i++) {
                auto f = pmp::Face(i);
                double ww = 0, w_s, w_r, w;
                new_Qf[f] = Quadric(0, 0, 0, 0);
                std::set<Face> neigh_f;
                for(Vertex v1: mesh_.vertices(f) )
                    for(Face vf: mesh_.faces(v1) )
                        neigh_f.insert(vf);

                for(auto ff: neigh_f) {
                    double d = sqrnorm(centroid(mesh_, ff) - centroid(mesh_, f));
                    w_s = std::exp(-0.5 * d / (sigma_s * sigma_s)) ;

                    double dis = quadric_distance(f_quadric_[f], ff);

                    w_r = std::exp(-0.5 * dis / (sigma_r * sigma_r)) ;
                    w = w_s * w_r;
                    Quadric _quad = f_quadric_[ff];
                    _quad *= w;
                    new_Qf[f] += _quad;
                    ww += w;
                }
                new_Qf[f] *= 1.0/ww;
            }}
            OMP_PARALLEL { OMP_FOR
                for(int i = 0; i < mesh_.n_faces(); i++){
                    auto f = pmp::Face(i);
                    f_quadric_[f] = new_Qf[f];
                }
            }
        }
        mesh_.remove_face_property(new_Qf);
    }

    Quadric compute_face_quadric(Face f) {
        return Quadric( f_normal_[f], mesh_.position( *mesh_.vertices(f) ) );
    }
    void compute_face_quadric(){
        for(auto f: mesh_.faces()) f_quadric_[f] = compute_face_quadric(f);
    }
    void compute_vertex_quadric(double sigma_b = 0.1){
        for(auto v: mesh_.vertices())
            v_quadric_[v] = compute_vertex_quadric(v, sigma_b);
    }

    Quadric compute_vertex_quadric(Vertex v, double sigma_b = 0.1) {
        double ww = 0, w;
        Quadric Q_v;
        for(Face f: mesh_.faces(v) ) {
            double d = sqrnorm(centroid(mesh_, f) - mesh_.position(v));
            w = std::exp(-0.5 * d / (sigma_b * sigma_b)) ;

            Quadric f_quadric = f_quadric_[f];
            f_quadric *= w;
            Q_v += f_quadric;
            ww += w;
        }
        Q_v *= 1/ww;
        return Q_v;
    }

private:
    //! the mesh
    SurfaceMesh& mesh_;
    FaceProperty<Quadric>       f_quadric_;
    FaceProperty<Normal>        f_normal_;
    VertexProperty<Quadric>     v_quadric_;
};

} // namespace pmp



using namespace pmp;



class Viewer : public MeshViewer
{
public:
    Viewer(const char* title, int width, int height);

protected:
    virtual void process_imgui();

private:
    SurfaceSmoothing smoother_;
};

Viewer::Viewer(const char* title, int width, int height)
    : MeshViewer(title, width, height), smoother_(mesh_)
{
    crease_angle_ = 180.0;
}

void Viewer::process_imgui()
{
    MeshViewer::process_imgui();

    ImGui::Spacing();
    ImGui::Spacing();

    if (ImGui::CollapsingHeader("Curvature", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (ImGui::Button("Mean Curvature"))
        {
            SurfaceCurvature analyzer(mesh_);
            analyzer.analyze_tensor(1, true);
            analyzer.mean_curvature_to_texture_coordinates();
            update_mesh();
            mesh_.use_cold_warm_texture();
            set_draw_mode("Texture");
        }
    }

    ImGui::Spacing();
    ImGui::Spacing();

    if (ImGui::CollapsingHeader("Smoothing", ImGuiTreeNodeFlags_DefaultOpen))
    {
        {
            static int n_iter = 1;

            static float sigma_b = 1;
            static float sigma_s = 2;
            static float sigma_r = 5;
            ImGui::PushItemWidth(100);
            ImGui::SliderFloat("sigma_b", &sigma_b, 0, 5);
            ImGui::PopItemWidth();
            ImGui::PushItemWidth(100);
            ImGui::SliderFloat("sigma_s", &sigma_s, 0, 5);
            ImGui::PopItemWidth();
            ImGui::PushItemWidth(100);
            ImGui::SliderFloat("sigma_r", &sigma_r, 0, 5);
            ImGui::PopItemWidth();
            ImGui::PushItemWidth(100);
            ImGui::SliderInt("n_iters", &n_iter, 1, 10);
            ImGui::PopItemWidth();

            if (ImGui::Button("QGFSmoothing"))
            {
                QGFSmoothing qgfSmoothing(mesh_);
                double mean_edge_l = qgfSmoothing.mean_edge_length() * 2;
                qgfSmoothing.do_smoothing(n_iter, mean_edge_l*sigma_b, mean_edge_l*sigma_s, mean_edge_l * sigma_r);
                update_mesh();
                mesh_.write("output1.off");
            }
        }
        static int weight = 0;
        ImGui::RadioButton("Cotan Laplace", &weight, 0);
        ImGui::RadioButton("Uniform Laplace", &weight, 1);
        bool uniform_laplace = (weight == 1);

        static int iterations = 10;
        ImGui::PushItemWidth(100);
        ImGui::SliderInt("Iterations", &iterations, 1, 100);
        ImGui::PopItemWidth();

        if (ImGui::Button("Explicit Smoothing"))
        {
            smoother_.explicit_smoothing(iterations, uniform_laplace);
            update_mesh();
        }

    }
}

int main(int argc, char** argv)
{
#ifndef __EMSCRIPTEN__
    Viewer window("Smoothing", 800, 600);
    if (argc == 2)
        window.load_mesh(argv[1]);
    return window.run();
#else
    Viewer window("Smoothing", 800, 600);
    window.load_mesh(argc == 2 ? argv[1] : "input.off");
    return window.run();
#endif
}
