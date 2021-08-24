// Copyright 2011-2019 the Polygon Mesh Processing Library developers.
// Distributed under a MIT-style license, see LICENSE.txt for details.

#include <pmp/visualization/MeshViewer.h>
#include <pmp/algorithms/SurfaceCurvature.h>
#include <pmp/algorithms/SurfaceSmoothing.h>
#include <imgui.h>
#include <pmp/algorithms/Quadric.h>
#include <pmp/algorithms/DifferentialGeometry.h>
#include <pmp/algorithms/SurfaceNormals.h>

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
        compute_vertex_quadric(s_b);
        for(int i = 0; i < iters; i++) {
            quadric_diffusion(10, s_s, s_r);
            quadric_motion(20);
        }
    }



private:

    void quadric_motion(int n_iter) {
        auto vnormal = mesh_.vertex_property<Normal>("v:normal");
        for(int i_iter = 0; i_iter < n_iter; i_iter++){
            SurfaceNormals::compute_vertex_normals(mesh_);
            for (auto v : mesh_.vertices()) {
                double lambda = v_quadric_[v].vertex_gradient(vnormal[v], mesh_.position(v));
                mesh_.position(v) += vnormal[v] * lambda;
            }
        }
    }

    void quadric_diffusion(int n_iter, double sigma_s = 0.1, double sigma_r = 0.1) {
        VertexProperty<Quadric> new_Qv = mesh_.vertex_property<Quadric>("f:quadric_");
        for(int i_iter = 0; i_iter < n_iter; i_iter++) {
            for(auto v: mesh_.vertices()) {
                double ww = 0, w_s, w_r, w;
                new_Qv[v] = Quadric(0, 0, 0, 0);
                for(Vertex vv: mesh_.vertices(v) ) {
                    double d = norm(mesh_.position(vv) - mesh_.position(v));
                    w_s = std::exp(-0.5 * d * d / (sigma_s * sigma_s)) ;
                    double qem = v_quadric_[vv](mesh_.position(v));
                    w_r = std::exp(-0.5 * qem * qem / (sigma_r * sigma_r)) ;
                    w = w_s * w_r * voronoi_area(mesh_, vv);
                    Quadric _quad = v_quadric_[vv];
                    _quad *= w;
                    new_Qv[v] += _quad;
                    ww += w;
                }
                new_Qv[v] *= 1.0/ww;
            }
            for(auto v: mesh_.vertices())
                v_quadric_[v] = new_Qv[v];
        }
        mesh_.remove_vertex_property(new_Qv);
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
            double d = norm(centroid(mesh_, f) - mesh_.position(v));
            w = std::exp(-0.5 * d * d / (sigma_b * sigma_b)) * triangle_area(mesh_, f);

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
            static int n_iter = 0;

            static float sigma_b = 0.1;
            static float sigma_s = 0.1;
            static float sigma_r = 0.1;
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

        ImGui::Spacing();
        ImGui::Spacing();

        static float timestep = 0.001;
        float lb = uniform_laplace ? 1.0 : 0.001;
        float ub = uniform_laplace ? 100.0 : 1.0;
        ImGui::PushItemWidth(100);
        ImGui::SliderFloat("TimeStep", &timestep, lb, ub);
        ImGui::PopItemWidth();

        if (ImGui::Button("Implicit Smoothing"))
        {
            // does the mesh have a boundary?
            bool has_boundary = false;
            for (auto v : mesh_.vertices())
                if (mesh_.is_boundary(v))
                    has_boundary = true;

            // only re-scale if we don't have a (fixed) boundary
            bool rescale = !has_boundary;

            Scalar dt =
                uniform_laplace ? timestep : timestep * radius_ * radius_;
            try
            {
                smoother_.implicit_smoothing(dt, uniform_laplace, rescale);
            }
            catch (const SolverException& e)
            {
                std::cerr << e.what() << std::endl;
                return;
            }
            update_mesh();
        }


        if (ImGui::Button("QGF"))
        {

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
