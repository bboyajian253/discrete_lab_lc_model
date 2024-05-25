# lifecycle_labor: parameters_shocks
# defines dictionary containing model parameters

import toolbox
import numpy as np
from scipy.stats import norm, uniform
from interpolation.splines import UCGrid  # see, https://github.com/EconForge/interpolation.py
from numba import b1, int64, float64
from numba.experimental import jitclass


HumanCapital_inputs = [
    ('J', int64),
    ('JR', int64),
    ('ω', float64),
    ('γ', float64),
    ('ψ', float64),
    ('β', float64),
    ('τ0', float64),
    ('τ1', float64),
    ('τc', float64),
    ('T', float64),
    ('ybar', float64),
    ('σ_me_n', float64),
    ('σ_me_e', float64),
    ('Nk', int64),
    ('Nsim', int64),
    ('Na', int64),
    ('n_indivs', int64),
    ('mink', float64),
    ('maxk', float64),
    ('minh', float64),
    ('maxh', float64),
    ('w_0', float64),
    ('r_0', float64),
    ('minc', float64),
    ('minn', float64),
    ('maxn', float64),
    ('model_mult_n', float64),
    ('model_mult_eph', float64),
    ('gridk', float64[:]),
    ('grida', float64[:, :]),
    ('shell_eval', float64[:]),
    ('shell_cc', float64[:]),
]

# noinspection NonAsciiCharacters
@jitclass(HumanCapital_inputs)
class Params_numba:
    def __init__(me,
                 grida,  # parameters for ability types
                 J=29,  # Number of periods of life
                 JR=22,  # Number of periods of working career
                 ω=1 - 1e-9,  # CRRA utility parameter
                 γ=0.30,  # Frisch elasticity; when value leisure, use γ=2
                 ψ=1.00,  # disutility of labor
                 β=1 / 1.02,  # Period utility discount rate
                 τ0=1.00,  # 1 - Level income tax parameter
                 τ1=0.00,  # Slope income tax parameter
                 τc=0.00,  # Consumption tax parameter
                 T=0.00,  # Lump sum gov't transfer
                 ybar=0.50,  # Avg. income (taxes set for y / ybar)
                 σ_me_n=1e-5,  # SD of iid hours measurement error shock
                 σ_me_e=1e-5,  # SD of iid earnings measurement error shock
                 Nk=301,  # Number of physical capital gridpoints
                 Nsim=100,  # Number of simulations for each (a, ψ) gridpoint
                 Na=3,  # number of learning ability gridpoints
                 mink=-15,  # minimum asset value
                 maxk=15,  # maximum asset value
                 minn=0.00,  # Min production time
                 maxn=0.80,  # Max production time
                 model_mult_n=100 * 50,  # Normalization to align model and data
                 model_mult_eph=26.5,  # Normalization to align model and data
                 ):

        # Store parameter values that are inputs to the class
        me.J, me.JR = J, JR
        me.ω, me.γ, me.ψ, me.β = ω, γ, ψ, β
        me.τ0, me.τ1, me.τc, me.T, me.ybar = τ0, τ1, τc, T, ybar
        me.σ_me_n, me.σ_me_e = σ_me_n, σ_me_e
        me.Nk, me.Nsim, me.Na = Nk, Nsim, Na
        me.n_indivs = me.Na * me.Nsim
        me.model_mult_n, me.model_mult_eph = model_mult_n, model_mult_eph
        me.grida = grida

        # Store other computational parameters
        me.minc = 1e-3
        me.mink, me.maxk = mink, maxk
        me.minn, me.maxn = minn, maxn

        # Store basic 1d grids
        me.gridk = np.linspace(me.mink, me.maxk, me.Nk)

        # store eval_linear evaluation-point-shells
        me.shell_eval = np.zeros(1)


# noinspection NonAsciiCharacters
class Params_nonumba:
    def __init__(me, par, path):
        me.par = par
        me.path = path

        # define other parameters
        me.age = np.arange(25, 80 + 1, 1)
        me.model_mult = {'n': par.model_mult_n, 'eph': par.model_mult_eph}
        me.model_mult['e'] = me.model_mult['eph'] * me.model_mult['n']
        for v in ['n', 'n_tru']:
            me.model_mult[v] = me.model_mult['n']

        # Store (non-jit-able) shapes for matrices
        me.shape_kk = (par.Na, par.Nk)
        me.shape_k = (par.Na, par.Nk)
        me.shape_j = (par.Na, par.Nk)
        me.shapestatesolns = (par.Na, par.Nk, par.J)
        me.shapesim = (par.Na, par.Nsim, par.J + 1)

        # Store (non-jit-able) grids for interpolation during model solution, simulation
        
        #Do we need the UCGrid() command here?
        me.grid_intrp_sim = UCGrid((par.gridk[0], par.gridk[-1], par.Nk))
        # i think this grid should be jit-able since its from a package that appears to be designed to be jit-able
        # may require @cuda.jit(debug=True) on UCGrid
        # dummy functions
        # does all this function do is check the validity of these numba types?
        # def UCGrid(*args):
        #     tt = numba.typeof((10.0, 1.0, 1))
        #     for a in args:
        #         assert numba.typeof(a) == tt
        #         min, max, n = a
        #         assert min < max
        #         assert n > 1
        #
        #        return tuple(args)
        # this also appears jitable as np.zeros is supported by numba
        me.sim_shell = np.zeros([par.Na, par.J + 1])

# noinspection NonAsciiCharacters
class Shocks:
    def __init__(me,
                 par,
                 par2
                 ):
        me.par = par
        me.par2 = par2

        # draw (Na, Nsim, J)-sequence of measurement error shocks for hours and earnings
        me.seed_me_n = 9283498
        merr_n = norm()
        merr_n.random_state = np.random.RandomState(seed=me.seed_me_n)
        N_shocks = np.prod(me.par2.shapesim)
        me.me_n = me.par.σ_me_n * merr_n.rvs(size=N_shocks).reshape(me.par2.shapesim)
        me.seed_me_e = 19872348
        merr_e = norm()
        merr_e.random_state = np.random.RandomState(seed=me.seed_me_e)
        me.me_e = me.par.σ_me_e * merr_e.rvs(size=N_shocks).reshape(me.par2.shapesim)
