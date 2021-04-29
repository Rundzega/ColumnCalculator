import math
from FCO import ElementosDaSecao
import matplotlib.pyplot as plt
import time


class SecaoTransversal(object):

    def __init__(self, list_retangulos, list_barras, classe_conc, classe_aco, nr_disc_x, nr_disc_y):
        self.area_composta = None  # cm2
        self.momento_estatico_x = None  # cm3
        self.momento_estatico_y = None  # cm3
        self.x_cg = None  # cm
        self.y_cg = None  # cm
        self.y_max = None  # cm
        self.y_min = None  # cm
        self.h = None  # cm
        self.d = None  # cm
        self.prof_lin_neut = None  # cm
        self.soma_nd = None  # cm
        self.mrd_x = None  # kncm
        self.mrd_y = None  # kncm
        self.envoltoria_momentos = []
        self.list_retangulos = list_retangulos  # lista de elementos Classe Retangulo
        self.list_barras = list_barras  # lista de elementos Classe BarraDeAco
        self.list_elem_disc = []  # lista de elementos Classe ElementosDiscretizados
        self.classe_conc = classe_conc  # Classe Concreto
        self.classe_aco = classe_aco  # Classe Aco
        self.inertia_x = 0  # cm4
        self.inertia_y = 0  # cm4
        self.n_elem_disc_x = nr_disc_x
        self.n_elem_disc_y = nr_disc_y
        self.rupturaNd = False
        self.get_centro_gravidade()
        self.discretizar_secao()

    def get_centro_gravidade(self):
        """Obtem o centro de gravidade de uma secao composta
        por retangulos
        """

        self.area_composta = 0
        self.momento_estatico_x = 0
        self.momento_estatico_y = 0

        for retangulo in self.list_retangulos:
            self.area_composta += retangulo.area
            self.momento_estatico_x += retangulo.momento_estatico_x
            self.momento_estatico_y += retangulo.momento_estatico_y

        self.x_cg = self.momento_estatico_y / self.area_composta
        self.y_cg = self.momento_estatico_x / self.area_composta

        for retangulo in self.list_retangulos:
            distancia_y = self.y_cg - retangulo.ret_y_cg
            distancia_x = self.x_cg - retangulo.ret_x_cg
            distancia_x_y = math.sqrt(distancia_x ** 2 + distancia_y ** 2)
            self.inertia_x += retangulo.inertia_x + \
                retangulo.area * (distancia_y ** 2)  # cm4
            self.inertia_y += retangulo.inertia_y + \
                retangulo.area * (distancia_x ** 2)  # cm4

    def discretizar_secao(self):
        """Cria os elementos da seção discretizada a partir dos retangulos definidos
        e do numero de elementos discretizados em x e y para cada retangulo
        """

        self.list_elem_disc.clear()
        for retangulo in self.list_retangulos:
            for i in range(self.n_elem_disc_x):
                for j in range(self.n_elem_disc_y):
                    xg_elem = retangulo.vertices[0][0] + i * retangulo.largura / self.n_elem_disc_x + retangulo.largura / (
                        2 * self.n_elem_disc_x)
                    yg_elem = retangulo.vertices[0][1] + j * retangulo.altura / self.n_elem_disc_y + retangulo.altura / (
                        2 * self.n_elem_disc_y)
                    self.list_elem_disc.append(ElementosDaSecao.ElementoDiscretizado(self.classe_conc, self.classe_aco, retangulo.largura / self.n_elem_disc_x,
                                                                                     retangulo.altura / self.n_elem_disc_y, xg_elem, yg_elem))

    def rotac_coord(self, coord, angulo):
        """Rotaciona um ponto de coordenadas x,y em função de um angulo alpha
        """

        rad = math.radians(angulo)
        new_pos = (coord[0] * math.cos(rad) + coord[1] * math.sin(rad), -
                   coord[0] * math.sin(rad) + coord[1] * math.cos(rad))
        return new_pos

    def get_parametros_rotac(self, angulo):
        """Obtem os parametros coordenada maxima da seção rotacionada,
        altura da seção rotacionada, e distancia d da borda mais comprimida
        até a armadura mais tracionada
        """

        verificacao_d = []
        vertices = []
        for retangulo in self.list_retangulos:
            for i in range(4):
                coord = (retangulo.vertices[i][0] - self.x_cg,
                         retangulo.vertices[i][1] - self.y_cg)
                v = self.rotac_coord(coord, angulo)
                vertices.append(v)

        self.y_max = 0
        self.y_min = 0
        for v in vertices:
            if v[1] > self.y_max:
                self.y_max = v[1]
            if v[1] < self.y_min:
                self.y_min = v[1]

        self.h = self.y_max - self.y_min

        for barra in self.list_barras:
            coord = (barra.x_pos - self.x_cg, barra.y_pos - self.y_cg)
            verificacao_d.append(self.rotac_coord(coord, angulo)[1])
        self.d = self.y_max - min(verificacao_d)

    def get_somatorio_nd(self, carga, linha_neutra, angulo):
        """Somatório de todas as forças normais atuantes na seção transveral, incluindo
        a solicitação nd, a partir de uma altura e angulo da linha neutra
        """

        self.soma_nd = 0

        # DOMINIO 2, ARAUJO
        if linha_neutra <= ((self.classe_conc.ecu / (self.classe_conc.ecu+self.classe_aco.esu)) * self.d):
            dom = 2

        elif linha_neutra <= self.h:  # DOMINIOS 3 E 4, ARAUJO
            dom = 3

        else:  # DOMINIO 5, ARAUJO
            dom = 5

        for elemento in self.list_elem_disc:
            pos = (elemento.elem_xcg - self.x_cg,
                   elemento.elem_ycg - self.y_cg)
            new_pos = self.rotac_coord(pos, angulo)
            elemento.get_def(new_pos[1], linha_neutra,
                             self.y_max, self.h, self.d, dom)
            elemento.get_tensao_conc()
            self.soma_nd -= elemento.tensao * elemento.area

        for barra in self.list_barras:
            posic_barra = (barra.x_pos - self.x_cg, barra.y_pos - self.y_cg)
            nova_pos = self.rotac_coord(posic_barra, angulo)
            barra.get_def(nova_pos[1], linha_neutra,
                          self.y_max, self.h, self.d, dom)
            barra.get_tensao_aco()
            barra.get_tensao_conc_equiv()
            self.soma_nd -= (barra.tensao -
                             barra.tensao_conc_equiv) * barra.area

        self.soma_nd -= carga
        return self.soma_nd

    def get_prof_ln(self, carga, tol, angulo):
        """Método da bisseção, varia a posição da LN
        para um angulo alpha e uma solicitação nd, para encontrar uma raiz para a função
        que equilibra as forças normais à seção
        """

        self.get_parametros_rotac(angulo)
        prof_min = -self.d
        prof_max = 2 * self.d
        prof_media = (prof_min + prof_max) / 2.0
        soma_min = self.get_somatorio_nd(carga, prof_min, angulo)
        soma_med = self.get_somatorio_nd(carga, prof_media, angulo)
        soma_max = self.get_somatorio_nd(carga, prof_max, angulo)
        counter = 1

        while soma_min * soma_max > 0:
            prof_min, prof_max = prof_max, prof_max * 10
            counter += 1
            if counter > 10:
                self.prof_lin_neut = False
                self.rupturaNd = True
                return
        else:
            while (prof_max - prof_min) / 2 > tol:
                counter = 0
                if soma_min * soma_med < 0:
                    prof_max = prof_media
                    prof_media = (prof_min + prof_max) / 2
                    soma_med = self.get_somatorio_nd(carga, prof_media, angulo)

                else:
                    prof_min = prof_media
                    prof_media = (prof_min + prof_max) / 2
                    soma_min = soma_med
                    soma_med = self.get_somatorio_nd(carga, prof_media, angulo)

            self.prof_lin_neut = prof_media

    def calc_momentos(self):
        """Calcula um par de momentos fletores Mxd e Myd em kNcm que leva à seção a ruina
        para determinado valor de esforço normal a partir de uma
        profundidade e orientação da linha neutra
        """

        self.mrd_x = 0
        self.mrd_y = 0
        if not self.prof_lin_neut:
            return

        for elemento in self.list_elem_disc:
            self.mrd_x -= (elemento.tensao * elemento.area *
                           (elemento.elem_xcg - self.x_cg))  # kncm
            self.mrd_y += (elemento.tensao * elemento.area *
                           (elemento.elem_ycg - self.y_cg))  # kncm

        for barra in self.list_barras:
            self.mrd_x -= ((barra.tensao - barra.tensao_conc_equiv)
                           * barra.area * (barra.x_pos - self.x_cg))  # kncm
            self.mrd_y += ((barra.tensao - barra.tensao_conc_equiv)
                           * barra.area * (barra.y_pos - self.y_cg))  # kncm

    def get_diagrama_n_mx_my(self, carga, tol, step_angulo_diag):
        """Varia a orientação da linha neutra em 360º a fim de obter uma envoltória
        pares de momentos fletores que levam a seção à ruina, a partir de uma força nd"""

        self.envoltoria_momentos.clear()
        for angle in range(0, 360, step_angulo_diag):
            self.get_prof_ln(carga, tol, angle)
            self.calc_momentos()
            if self.mrd_y == 0:  # SEM LN
                if self.mrd_x > 0:
                    angulo_grafico = 90
                elif self.mrd_x < 0:
                    angulo_grafico = 270
                else:
                    angulo_grafico = 0
            elif (self.mrd_y > 0 and self.mrd_x > 0):  # PRIMEIRO QUADRANTE
                angulo_grafico = math.degrees(
                    math.atan(abs(self.mrd_x) / abs(self.mrd_y)))

            elif (self.mrd_y < 0 and self.mrd_x >= 0):  # SEGUNDO QUADRANTE
                angulo_grafico = 180 - \
                    math.degrees(math.atan(abs(self.mrd_x) / abs(self.mrd_y)))

            elif (self.mrd_y < 0 and self.mrd_x < 0):  # TERCEIRO QUADRANTE
                angulo_grafico = 180 + \
                    math.degrees(math.atan(abs(self.mrd_x) / abs(self.mrd_y)))

            else:  # QUARTO QUADRANTE
                angulo_grafico = 360 - \
                    math.degrees(math.atan(abs(self.mrd_x) / abs(self.mrd_y)))

            self.envoltoria_momentos.append(
                (self.mrd_y/100, self.mrd_x/100, angulo_grafico))

    def get_esforcos_internos(self, deform_axial, curvatura_y, curvatura_x, fluen=True):
        """Obtém o esforço normal Nd, Momento fletor Y e Momento fletor X de uma seção
        a partir dos valores da deformação axial, curvatura_y, curvatura_x
        """
        nd = 0  # kn
        my = 0  # kncm
        mx = 0  # kncm

        for elemento in self.list_elem_disc:
            pos = (elemento.elem_xcg - self.x_cg,
                   elemento.elem_ycg - self.y_cg)
            elemento.deformacao = deform_axial + \
                pos[1] * curvatura_y/100 + pos[0] * curvatura_x/100
            elemento.get_tensao_conc(fluencia=fluen)
            nd += elemento.tensao * elemento.area
            my += (elemento.tensao * elemento.area *
                   (elemento.elem_ycg - self.y_cg))
            mx += (elemento.tensao * elemento.area *
                   (elemento.elem_xcg - self.x_cg))

        for barra in self.list_barras:
            posic_barra = (barra.x_pos - self.x_cg, barra.y_pos - self.y_cg)
            barra.deformacao = deform_axial + \
                posic_barra[1] * curvatura_y/100 + \
                posic_barra[0] * curvatura_x/100
            barra.get_tensao_aco()
            barra.get_tensao_conc_equiv(fluencia=fluen)
            nd += (barra.tensao - barra.tensao_conc_equiv) * barra.area
            my += ((barra.tensao - barra.tensao_conc_equiv)
                   * barra.area * (barra.y_pos - self.y_cg))
            mx += ((barra.tensao - barra.tensao_conc_equiv)
                   * barra.area * (barra.x_pos - self.x_cg))

        return nd, my/100, mx/100

    def check_ruptura(self):
        """Verifica se houve ruptura por esmagamento no concreto 
        ou alongamento excessivo da armadura de aço"""

        for elemento in self.list_elem_disc:
            if elemento.rompeu == True:
                return True
        for barra in self.list_barras:
            if barra.rompeu == True:
                return True
        return False
