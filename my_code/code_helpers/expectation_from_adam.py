# # Interpolate to find cc given (kk, hh, ππ)
# eval = par.shell_eval
# cc = par.shell_cc
# for nzz in range(par.Nz):
#     for nππ in range(par.n_π):
#         n_eval = nzz * par.n_π + nππ
#         eval[0] = hh * gridz_kron[n_eval]
#         cc[n_eval] = interp(par.gridh, mcc_flat[:, nππ], eval[0])

# # Compute implied c given cc
# dVV_dkk = np.sum(probzπ * (1 + r) * model.du_dc(cc, par))
# c = model.invert_c(dVV_dkk, par)
