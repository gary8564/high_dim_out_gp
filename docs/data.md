# Data Usage 

## Toy example: Environmental spill function

The Environmental Model function describes the concentration of a pollutant released by two spills in a 1D channel as a function of space and time. Let $C(\mathbf{x})$ denote the pollutant concentration at position $s$ and time $t$. Two equal-mass spills occur at space–time locations (0, 0) and ($L$, $\tau$). The model output is a $d_s$ × $d_t$ array containing scaled concentrations $C(s_i, t_j)$ evaluated on a space–time grid.

**Mathematical formulation**:

$$
f(\mathbf{x}) = \sqrt{4\pi} C(\mathbf{x}) \\
\text{where  } C(\mathbf{x}) = \frac{M}{\sqrt{4\pi D t}} \, \exp\!\left( -\frac{s^{2}}{4 D t} \right)
\; + \; I(t>\tau) \, \frac{M}{\sqrt{4\pi D (t-\tau)}} \, \exp\!\left( -\frac{(s- L)^{2}}{4 D (t-\tau)} \right)
$$

Here, $I(\cdot)$ is the indicator function and $0 \le s \le 3,\; t>0$. Inputs are $M$ (mass), $D$ (diffusion rate), $L$ (location of second spill), and $\tau$ (time of second spill).

**Input ranges and descriptions**:

| Parameter | Range            | Description                                   |
| --------- | ---------------- | --------------------------------------------- |
| $M$       | [7, 13]          | mass of pollutant spilled at each location    |
| $D$       | [0.02, 0.12]     | diffusion rate in the channel                 |
| $L$       | [0.01, 3]        | location of the second spill                  |
| $\tau$    | [30.01, 30.295]  | time of the second spill                      |

**Output**: $d_s$ × $d_t$ grid of concentrations

Source: [Environmental Model function (Virtual Library of Simulation Experiments)](https://www.sfu.ca/~ssurjano/environ.html). See also [Bliznyuk et al. (2008)](https://www.jstor.org/stable/27594307) for more theoretical background information.

## Case study A: Synthetic dataset

Topography consists of a parabolic slope starting at $x = 0$ at an altitude of $1332\ \mathrm{m}$, and connecting to a flat land at $x = 3000$ at an altitude of $0\ \mathrm{m}$. Extent of the area in x and y directions are $5000\ \mathrm{m}$ and $4000\ \mathrm{m}$, respectively, whereas the resolution is $20\ \mathrm{m}$. Release zone was defined as an elliptic cylinder, of which the centre is located at $(x, y) = (600, 2000)$ with a minor axis of $100\ \mathrm{m}$ and a major axis of $200\ \mathrm{m}$. Height at each point within the ellipse was defined as $20 m$, which generates a total release volume of $1.432 \times 10^6\ \mathrm{m}^3$. 

Source: [Yildiz, Anil et. al.](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2022.1032438/full)

## Case study B: Acheron Rock Avalanche

The Acheron rock avalanche—located near Canterbury, New Zealand—occurred approximately 1,100 years before present and was likely triggered by a strong earthquake. The deposit area was estimated to be 0.72 × 10<sup>6</sup> m<sup>2</sup> and the deposit volume roughly 8.9 × 10<sup>6</sup> m<sup>3</sup>.

The input/output dataset are obtained run via r.avaflow 2.3 with 100 simulations as training set and 20 additional simulations as test set. The input variables are dry-coulomb friction coefficient(μ), turbulent friction coefficient(ζ), release volume(ν₀), respectively. The output variables contain impact area, deposit area, deposit volume, maximum flow velocity, and maximum flow height at position (x, y). 

Source: [Yildiz, Anil et. al.](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2022.1032438/full)