## How much memory?
We can estimate the amount of memory required for the simulations considering the
biggest struct `Run`, i.e. the states `Started` and `Ended`.

### Considering `Started`
The main thing is `EcDNADistributionNPlus` which is a vec of `DNACopy` (`u16`).
Given a number of iterations n, we have: 16 bit * cells * 0.125 byte/bit, where
cells is the number of nplus cells. The number of nplus cells depends on the
proliferation rate of the cells w/ ecDNA $$\rho_1$$ strenght.

If $$\rho_1$$ is low (around 1), then cells will be approximately 0.1 * n, so the
memory required for one run will be approximately: 16 * 0.1 * n * 0.125 = 0.2n.

If $$\rho_1$$ is higher than 1.4, then cells will be approximately 0.1 * n, so
the memory required in the worst-case scenario for one run will be
approximately: 16 * n * 0.125 = 2n.


### Considering `Ended`
Once the run has been simulated, its state switches to `Ended` and the data in
`Started` is summarized by `Ended`. The biggest struct here is `EcDNADistribution`,
which I think should be negigeable compared to `EcDNADistributionNPlus` since it's
a sparse histogram with the number of entries being the number of k copies present
in the data.

### Dynamics settings
In the dynamics settings, we should add also the dynamical measurements, that keep
track of the state of the system e.g. the mean, nplus cells count per iteration...
This depends on the dynamics that we are simulating, e.g. mean is a vec of `f32`
wherease nplus is a vec of `u64`.
I think we can say that we can add to the previous estimations n bytes.

**Example: nplus/nminus.** u64 * n * timepoints * 0.125 bytes/bit = 64 bit * 0.125 bytes/bit * n * timepoints = 8 * bytes * n * timepoints.
