# Data Usage 

## Toy example: Environmental spill function

## Case study A: Synthetic dataset

Topography consists of a parabolic slope starting at $x = 0$ at an altitude of $1332\ \mathrm{m}$, and connecting to a flat land at $x = 3000$ at an altitude of $0\ \mathrm{m}$. Extent of the area in x and y directions are $5000\ \mathrm{m}$ and $4000\ \mathrm{m}$, respectively, whereas the resolution is $20\ \mathrm{m}$. Release zone was defined as an elliptic cylinder, of which the centre is located at $(x, y) = (600, 2000)$ with a minor axis of $100\ \mathrm{m}$ and a major axis of $200\ \mathrm{m}$. Height at each point within the ellipse was defined as $20 m$, which generates a total release volume of $1.432 \times 10^6\ \mathrm{m}^3$. 

The dataset is taken from [Yildiz, Anil et. al.](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2022.1032438/full)

## Case study B: Acheron Rock Avalanche

The Acheron rock avalanche—located near Canterbury, New Zealand—occurred approximately 1,100 years before present and was likely triggered by a strong earthquake. The deposit area was estimated to be 0.72 × 10<sup>6</sup> m<sup>2</sup> and the deposit volume roughly 8.9 × 10<sup>6</sup> m<sup>3</sup>.

The input/output dataset can be found in [Yildiz, Anil et. al.](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2022.1032438/full) with 100 simulations run via r.avaflow 2.3 as training set and 20 additional simulations as test set. The input variables are dry-coulomb friction coefficient(μ), turbulent friction coefficient(ζ), release volume(ν₀), respectively. The output variables contain impact area, deposit area, deposit volume, maximum flow velocity, and maximum flow height at position (x, y). 
