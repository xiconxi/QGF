A demo for [Quadric-based Mesh Denoising](https://perso.telecom-paristech.fr/boubek/papers/QGF/QGF_lowres.pdf) and [probabilistic-quadrics](https://www.graphics.rwth-aachen.de/media/papers/308/probabilistic-quadrics.pdf)

See also. [github:probabilistic-quadrics](https://github.com/Philip-Trettner/probabilistic-quadrics)



--- 


## Getting Started

Clone the repository:

```sh
git clone --recursive https://github.com/pmp-library/pmp-library.git
```

Configure and build:

```sh
cd pmp-library && mkdir build && cd build && cmake .. && make
```

Run the quadric smoothing app:

```sh
./QGF /path/to/mesh.off
```
or
```sh
./PQGF /path/to/mesh.off
```
