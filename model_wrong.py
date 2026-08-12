import torch
import matplotlib.pyplot as plt
if __name__ == "__main__":

    OPNO_darcy_neumann = [0.013205704931169748, 0.004439614603761584, 0.0009870246215723454, 0.0011640325968619435]
    OPNO_darcy_neumann_a = [0.07223663292825222, 0.020319873448461295, 0.01398596614599228, 0.012769257430918515]
    OPNO_helmholtz = [0.5275152921676636, 0.21131119549274444, 0.013945425692945719, 0.007474528094753623]
    OPNO_burger = [0.0360494189336896, 0.012313122395426035, 0.007180788363330066, 0.005975150992162526]

    
    UN_darcy_neumann_a = [0.06468855546787382, 0.030211953353136776, 0.017946711499243973, 0.013101805555634201]
    UN_burger = [0.029944943990558387, 0.011413796339184046, 0.008361507765948772, 0.0067607883363962175]
    UN_helmholtz = [0.09747105143964291, 0.017240127976983786, 0.010381131432950497, 0.007538055898621678]
    UN_darcy_neumann = [0.0009124755102675408, 0.0006669286522082984, 0.0006276480661472306, 0.0005958653386915103]

    def rel_loss(list1, list2):
        return [i/j for i,j in zip(list1, list2)]
    L = [1,2,4,8]

    linewidth=4
    markersize=11
    fontsize=20
    plt.figure(figsize=(12, 10))
    plt.semilogy(L, rel_loss(OPNO_darcy_neumann_a, UN_darcy_neumann_a), 'ro:', linewidth=linewidth, markersize=markersize, label=r'Darcy $a \rightarrow u$')
    plt.semilogy(L, rel_loss(OPNO_helmholtz, UN_helmholtz), 'ko:', linewidth=linewidth, markersize=markersize, label=r'Helmholtz $f \rightarrow u$')
    plt.semilogy(L, rel_loss(OPNO_darcy_neumann, UN_darcy_neumann), 'go:', linewidth=linewidth, markersize=markersize, label=r'Darcy $f \rightarrow u$')
    plt.semilogy(L, rel_loss(OPNO_burger, UN_burger), 'bo:', linewidth=linewidth, markersize=markersize, label=r'Burgers $u_0 \rightarrow u$')
    plt.legend(fontsize=fontsize)
    plt.xticks(L)
    plt.tick_params(labelsize=fontsize)
    plt.xlabel(r'$n_{layers}$', fontsize=fontsize)
    plt.ylabel(r'$\frac{E_{OPNO}}{E_{UltraNet}}$', fontsize=2*fontsize)
    # plt.semilogy(L, OPNO_darcy_neumann_a, 'bP:', linewidth=linewidth, markersize=markersize)
    # plt.semilogy(L, UN_burger, 'bo--', linewidth=linewidth, markersize=markersize)
    # plt.semilogy(L, OPNO_helmholtz, 'kP:', linewidth=linewidth, markersize=markersize)
    # plt.semilogy(L, UN_helmholtz, 'ko--', linewidth=linewidth, markersize=markersize)
    plt.show()