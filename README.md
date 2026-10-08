# dark-forest

![Dark forest branches blending into a cosmic web of dark-matter halos](docs/assets/dark-forest-banner.png)

This code uses Monte-Carlo algorithm to generate merger trees for cosmological dark matter N-body simulations.

The code is able to evolve the position and velocity of the Monte-Carlo haloes as well.

Dark Forest was also used in [Balu et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.520.3368B/abstract) and follow-up studies of galaxy formation and reionization. In this application, the merger trees of the L210 simulation, a box of side length 210 h⁻¹ Mpc, were augmented to include haloes down to approximately 2 × 10⁷ h⁻¹ M☉, resolving a statistically complete sample of atomic-cooling haloes down from $z=20$.

Original author: [Yisheng Qiu](https://github.com/yqiuu)

Maintainer: [Balu Sreedhar](https://github.com/s-balu)

For issues or questions, contact [bsreedhar@us.es](mailto:bsreedhar@us.es).

If you use this code, cite [Qiu et al. 2020](https://ui.adsabs.harvard.edu/abs/2020MNRAS.500..493Q/abstract):

```
@ARTICLE{2020MNRAS.500..493Q,
       author = {{Qiu}, Yisheng and {Mutch}, Simon J. and {Elahi}, Pascal J. and
         {Poulton}, Rhys J.~J. and {Power}, Chris and {Wyithe}, J. Stuart B.},
        title = "{An efficient hybrid method to produce high-resolution large-volume dark matter simulations for semi-analytic models of reionization}",
      journal = {\mnras},
     keywords = {methods: numerical, galaxies: high-redshift, dark ages, reionization, first stars, Astrophysics - Astrophysics of Galaxies, Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2020,
        month = oct,
       volume = {500},
       number = {1},
        pages = {493-505},
          doi = {10.1093/mnras/staa3261},
archivePrefix = {arXiv},
       eprint = {2007.14624},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020MNRAS.500..493Q},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

If using the L210 augmentation and associated datasets, also cite [Balu et al. 2023](https://ui.adsabs.harvard.edu/abs/2023MNRAS.520.3368B/abstract):

```bibtex
@ARTICLE{2023MNRAS.520.3368B,
       author = {{Balu}, Sreedhar and {Greig}, Bradley and {Qiu}, Yisheng and
         {Power}, Chris and {Qin}, Yuxiang and {Mutch}, Simon and
         {Wyithe}, J. Stuart B.},
        title = "{Thermal and reionization history within a large-volume semi-analytic galaxy formation simulation}",
      journal = {Monthly Notices of the Royal Astronomical Society},
         year = 2023,
       volume = {520},
       number = {3},
        pages = {3368--3382},
          doi = {10.1093/mnras/stad281},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023MNRAS.520.3368B}
}
```
