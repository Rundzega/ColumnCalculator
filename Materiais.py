import math


class Concreto:

    """Classe do material concreto"""

    def __init__(self, fck, gamma_c, coef_fluencia, alpha, beta):  # fck em MPa

        self.fck = fck/10  # kN/cm2
        self.gamma_c = gamma_c
        self.fcd = self.fck/gamma_c  # kN/cm2
        self.fluenc_efetiva = 0.57 * coef_fluencia
        self.ecu = 0.0035 if fck <= 50 else 0.0026 + 0.035 * \
            ((90 - fck) / 100) ** 4   # limites de deformacao concreto
        self.eco = 0.002 if fck <= 50 else 0.002 + \
            (0.085 / 1000) * (fck - 50) ** 0.53  # limites de deformacao concreto
        self.ecu_f = self.ecu * (1 + self.fluenc_efetiva)
        self.eco_f = self.eco * (1 + self.fluenc_efetiva)

        modulo_e = (21500 * ((fck/10) ** (1/3)))  # MPa
        ecd = modulo_e/1.2
        n = (gamma_c * ecd * self.eco)/(alpha*fck)
        self.n = n if n <= 2 else 2  # parametro para diagrama deformacao conc
        self.alpha = alpha
        self.beta = beta
        self.modulo_e = modulo_e/10  # kn/cm2
        self.e_co = self.eco * \
            (1 - (1 - (self.beta / self.alpha) ** (1 / self.n)))
        self.e_co_f = self.eco_f * \
            (1 - (1 - (self.beta / self.alpha) ** (1 / self.n)))


class Aco:

    """Classe do material aço"""

    def __init__(self, fyk, gamma_s, es, esu):  # fyk em MPa, es em MPa

        self.fyk = fyk/10  # kN/cm2
        self.fyd = self.fyk/gamma_s  # kN/cm2
        self.Es = es / 10  # kN/cm2
        self.esu = esu
