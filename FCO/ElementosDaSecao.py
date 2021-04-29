import numpy as np


class BarraDeAco(object):  # Adiciona uma nova barra

    """Elemento barra de aço da seção transversal"""

    def __init__(self, classe_conc, classe_aco, diametro, x_pos, y_pos):
        self.diametro = diametro  # mm
        self.x_pos = x_pos  # cm
        self.y_pos = y_pos  # cm
        self.area = (np.pi*(self.diametro**2)/4)/100  # cm2
        self.tensao = None  # kn/cm2
        self.tensao_conc_equiv = None  # kn/cm2

        self.classe_conc = classe_conc
        self.classe_aco = classe_aco
        self.rompeu = False

    def get_def(self, y_linha, linha_neutra, y_max, h, d, dom):
        """Obtém a deformação específica
        em uma ponto situado a uma distância y da linha neutra
        no dominio 2
        """
        di = y_max - y_linha
        if dom == 2:
            self.deformacao = - (self.classe_aco.esu *
                                 ((linha_neutra - di) / (d - linha_neutra)))
        elif dom == 3 or dom == 4:
            self.deformacao = - (self.classe_conc.ecu *
                                 ((linha_neutra - di) / linha_neutra))
        else:
            self.deformacao = - (self.classe_conc.eco * ((linha_neutra - di) / (
                linha_neutra - (1 - self.classe_conc.eco / self.classe_conc.ecu) * h)))

    def get_tensao_aco(self):  # tensao na barra de aço em kN/cm²
        """Obtem a tensão em kN/cm² em uma barra de aço submetida
        a uma deformacao
        """

        if self.deformacao <= self.classe_aco.esu:

            self.tensao = self.classe_aco.Es * self.deformacao
            if self.tensao >= 0:
                if self.tensao >= self.classe_aco.fyd:
                    self.tensao = self.classe_aco.fyd
            if self.tensao <= 0:
                if self.tensao <= -self.classe_aco.fyd:
                    self.tensao = -self.classe_aco.fyd

        else:
            self.rompeu = True
            self.tensao = 0

    def get_tensao_conc_equiv(self, fluencia=False):
        """Obtem a tensão de compressão em kN/cm² em um elemento discretizado
        de concreto submetido a uma deformacao Diagrama parabola retangulo
        """

        if fluencia:
            if self.deformacao < 0:
                if abs(self.deformacao) < self.classe_conc.e_co_f:
                    self.tensao_conc_equiv = - (self.classe_conc.alpha * self.classe_conc.fcd * (
                        1 - (1 - abs(self.deformacao) / self.classe_conc.eco_f) ** self.classe_conc.n))
                elif abs(self.deformacao) <= self.classe_conc.ecu_f:
                    self.tensao_conc_equiv = - \
                        (self.classe_conc.beta * self.classe_conc.fcd)
                else:
                    self.tensao_conc_equiv = 0
            else:
                self.tensao_conc_equiv = 0

        else:
            if self.deformacao < 0:
                if abs(self.deformacao) < self.classe_conc.e_co:
                    self.tensao_conc_equiv = - (self.classe_conc.alpha * self.classe_conc.fcd * (
                        1 - (1 - abs(self.deformacao) / self.classe_conc.eco) ** self.classe_conc.n))
                elif abs(self.deformacao) <= self.classe_conc.ecu:
                    self.tensao_conc_equiv = - \
                        (self.classe_conc.beta * self.classe_conc.fcd)
                else:
                    self.tensao_conc_equiv = 0
            else:
                self.tensao_conc_equiv = 0


class ElementoDiscretizado(object):  # Elementos discretizados de concreto

    """Elemento retangular da seção transversal de concreto armado discretizada"""

    def __init__(self, classe_conc, classe_aco, largura, altura, elem_xcg, elem_ycg):
        self.elem_xcg = elem_xcg  # cm
        self.elem_ycg = elem_ycg  # cm
        self.area = largura * altura  # cm2
        self.largura = largura  # cm
        self.altura = altura  # cm
        self.vertices = np.array([
            (elem_xcg - largura/2, elem_ycg - altura/2),
            (elem_xcg + largura/2, elem_ycg - altura/2),
            (elem_xcg + largura/2, elem_ycg + altura/2),
            (elem_xcg - largura/2, elem_ycg + altura/2)
        ])
        self.tensao = None  # kn/cm2
        self.classe_conc = classe_conc
        self.classe_aco = classe_aco
        self.rompeu = False

    def get_def(self, y_linha, linha_neutra, y_max, h, d, dom):
        """Obtém a deformação específica
        em uma ponto situado a uma distância y da linha neutra
        no dominio 2
        """

        di = y_max - y_linha
        if dom == 2:
            self.deformacao = - (self.classe_aco.esu *
                                 ((linha_neutra - di) / (d - linha_neutra)))
        elif dom == 3 or dom == 4:
            self.deformacao = - (self.classe_conc.ecu *
                                 ((linha_neutra - di) / linha_neutra))
        else:
            self.deformacao = - (self.classe_conc.eco * ((linha_neutra - di) / (
                linha_neutra - (1 - self.classe_conc.eco / self.classe_conc.ecu) * h)))

    def get_tensao_conc(self, fluencia=False):
        """Obtem a tensão de compressão em kN/cm² em um elemento discretizado
        de concreto submetido a uma deformacao Diagrama parabola retangulo 'oade' Araújo
        """

        if fluencia:
            if self.deformacao < 0:
                if abs(self.deformacao) < self.classe_conc.e_co_f:
                    self.tensao = - (self.classe_conc.alpha * self.classe_conc.fcd * (
                        1 - (1 - abs(self.deformacao) / self.classe_conc.eco_f) ** self.classe_conc.n))
                elif abs(self.deformacao) <= self.classe_conc.ecu_f:
                    self.tensao = - (self.classe_conc.beta *
                                     self.classe_conc.fcd)
                else:
                    self.tensao = 0
            else:
                self.tensao = 0

        else:
            if self.deformacao < 0:
                if abs(self.deformacao) < self.classe_conc.e_co:
                    self.tensao = - (self.classe_conc.alpha * self.classe_conc.fcd * (
                        1 - (1 - abs(self.deformacao) / self.classe_conc.eco) ** self.classe_conc.n))
                elif abs(self.deformacao) <= self.classe_conc.ecu:
                    self.tensao = - (self.classe_conc.beta *
                                     self.classe_conc.fcd)
                else:
                    self.tensao = 0
                    self.rompeu = True
            else:
                self.tensao = 0


class Retangulo:  # Retangulo que compoe seção de concreto

    """Elemento retangular de concreto armado para composição da seção transversal"""

    def __init__(self, largura, altura, ret_x_cg, ret_y_cg):
        self.largura = largura  # cm
        self.altura = altura  # cm
        self.ret_x_cg = ret_x_cg  # cm
        self.ret_y_cg = ret_y_cg  # cm
        self.area = largura * altura  # cm2
        self.vertices = np.array([
            (ret_x_cg - largura/2, ret_y_cg - altura/2),
            (ret_x_cg + largura/2, ret_y_cg - altura/2),
            (ret_x_cg + largura/2, ret_y_cg + altura/2),
            (ret_x_cg - largura/2, ret_y_cg + altura/2)
        ])
        self.momento_estatico_x = self.area*ret_y_cg  # cm3
        self.momento_estatico_y = self.area*ret_x_cg  # cm3
        self.inertia_x = (self.largura * (self.altura ** 3))/12  # cm4
        self.inertia_y = ((self.largura ** 3) * self.altura)/12  # cm4
