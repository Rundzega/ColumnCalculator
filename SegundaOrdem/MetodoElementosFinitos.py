import numpy as np
import matplotlib.pyplot as plt
import time


class NodeFEM:

    """NÓ do modelo de elementos finitos"""

    def __init__(self, index, z_pos):

        self.index = index
        self.z_pos = z_pos
        self.nodal_restrictions = np.ones(5)  # 0 fixo, 1 livre
        self.nodal_loads = np.zeros(5)  # kn, kn, kncm, kn, kncm
        self.deslocamentos = np.zeros(5)  # cm


class BarElementFEM:

    """Barra do modelo de elementos finitos"""

    def __init__(self, i_node, j_node, length, index, secao_transversal):

        self.i_node = i_node  # Classe NodeFEM
        self.j_node = j_node  # Classe NodeFEM
        self.secao_transversal = secao_transversal  # Classe SecaoTransversal
        self.length = length  # cm
        self.index = index
        self.deslocamentos_nodais = np.zeros(10)  # cm
        self.matriz_rigidez = None
        # cargas ao longo da barra, z(peso proprio), yb, yt, xb, xt #kn/cm
        self.linear_loads = np.zeros(5)
        self.equiv_nodal_loads = None
        self.non_linear_loads = np.zeros(10)
        self.linear_forces_on_ends = None

    def get_elem_stiff_mat(self):
        """OBTEM A MATRIZ DE RIGIDEZ 'K' DA BARRA, QUE LINEARIZA O SISTEMA DE EQUAÇÕES
        DAS FORÇAS E DESLOCAMENTOS"""

        modulo = self.secao_transversal.classe_conc.modulo_e * 100**2
        inercia_x = self.secao_transversal.inertia_x / (100**4)
        inercia_y = self.secao_transversal.inertia_y / (100**4)
        area = self.secao_transversal.area_composta / (100**2)

        c1 = modulo*area/self.length
        c2 = 12*modulo*inercia_x/self.length ** 3
        c3 = c2*self.length/2
        c4 = c2*self.length**2/3
        c5 = c4/2
        c6 = 12*modulo*inercia_y/self.length ** 3
        c7 = c6*self.length/2
        c8 = c6*self.length ** 2/3
        c9 = c8/2

        self.matriz_rigidez = np.array([[c1, 0, 0, 0, 0, -c1, 0, 0, 0, 0],
                                        [0, c2, c3, 0, 0, 0, -c2, c3, 0, 0],
                                        [0, c3, c4, 0, 0, 0, -c3, c5, 0, 0],
                                        [0, 0, 0, c6, c7, 0, 0, 0, -c6, c7],
                                        [0, 0, 0, c7, c8, 0, 0, 0, -c7, c9],
                                        [-c1, 0, 0, 0, 0, c1, 0, 0, 0, 0],
                                        [0, -c2, -c3, 0, 0, 0, c2, -c3, 0, 0],
                                        [0, c3, c5, 0, 0, 0, -c3, c4, 0, 0],
                                        [0, 0, 0, -c6, -c7, 0, 0, 0, c6, -c7],
                                        [0, 0, 0, c7, c9, 0, 0, 0, -c7, c8]])

    def get_equiv_nodal_loads(self):
        """Transforma as cargas ditribuidas ao longo do elemento em cargas nodais equivalentes"""

        z_load = self.linear_loads[0]
        coeficientes_y = np.array([self.linear_loads[1], self.linear_loads[2]])
        coeficientes_x = np.array([self.linear_loads[3], self.linear_loads[4]])
        pontos_comprimento = np.array([0, self.length])
        ay, by = np.polyfit(pontos_comprimento, coeficientes_y, deg=1)
        ax, bx = np.polyfit(pontos_comprimento, coeficientes_x, deg=1)

        """Integral de uma função pelo método de gauss legendre"""

        [t, w] = np.polynomial.legendre.leggauss(3)
        j = 0.5 * (self.length)
        x = np.array([(j * i + j) for i in t])

        self.equiv_nodal_loads = np.zeros(10)
        for n in range(10):
            if n == 0 or n == 5:
                self.equiv_nodal_loads[n] = j * \
                    np.sum(w * (z_load * self.shape_functions(n, x)))

            if n == 1 or n == 2 or n == 6 or n == 7:
                self.equiv_nodal_loads[n] = j * np.sum(
                    w * (self.triangular_loads_functions(x, ay, by) * self.shape_functions(n, x)))

            if n == 3 or n == 4 or n == 8 or n == 9:
                self.equiv_nodal_loads[n] = j * np.sum(
                    w * (self.triangular_loads_functions(x, ax, bx) * self.shape_functions(n, x)))

    def triangular_loads_functions(self, z, a, b):
        return a * z + b

    def shape_functions(self, elem_dof_index, z):
        """Retorna as Funções de forma para interpolação dos deslocamentos ao
        longo do elemento em funcao dos deslocamentos nodais, os graus
        de liberdade são numerados de 0 a 9:
        0: deslocamento axial (z) no nó inicial
        1: deslocamento transversal no plano zy do nó inicial
        2: rotação do nó inicial no plano zy
        3: deslocamento transversal no plano zx do nó inicial
        4: rotação do nó inicial no plano zx
        5: deslocamento axial (z) no nó final
        6: deslocamento transversal no plano zy do nó inicial
        7: rotação do nó final no plano zy
        8: deslocamento transversal no plano zx do nó final
        9: rotação do nó final no plano zx"""

        if elem_dof_index == 0:
            return 1 - z/self.length

        if elem_dof_index == 1 or elem_dof_index == 3:
            return 2 * (z/self.length) ** 3 - 3 * (z/self.length) ** 2 + 1

        if elem_dof_index == 2 or elem_dof_index == 4:
            return self.length * ((z/self.length) ** 3 - 2 * (z/self.length) ** 2 + (z/self.length))

        if elem_dof_index == 5:
            return (z/self.length)

        if elem_dof_index == 6 or elem_dof_index == 8:
            return - 2 * (z/self.length) ** 3 + 3 * (z/self.length) ** 2

        if elem_dof_index == 7 or elem_dof_index == 9:
            return self.length * ((z/self.length) ** 3 - (z/self.length) ** 2)

    def deriv_shape_functions(self, elem_dof_index, z):
        """Retorna as derivadas das Funções de forma para interpolação dos deslocamentos ao
        longo do elemento em funcao dos deslocamentos nodais, os graus
        de liberdade são numerados de 0 a 9:"""

        if elem_dof_index == 0:
            return - 1 / self.length

        if elem_dof_index == 1 or elem_dof_index == 3:
            return (6*z*(z-self.length))/(self.length ** 3)

        if elem_dof_index == 2 or elem_dof_index == 4:
            return (self.length ** 2 - 4 * self.length * z + 3*z**2)/(self.length ** 2)

        if elem_dof_index == 5:
            return 1/self.length

        if elem_dof_index == 6 or elem_dof_index == 8:
            return (6*z*(self.length-z))/(self.length**3)

        if elem_dof_index == 7 or elem_dof_index == 9:
            return (z * (3*z-2*self.length))/(self.length**2)

    def second_deriv_shape_functions(self, elem_dof_index, z):
        """Retorna as derivadas segundas das Funções de forma para interpolação dos deslocamentos ao
        longo do elemento em funcao dos deslocamentos nodais, os graus
        de liberdade são numerados de 0 a 9:"""

        if elem_dof_index == 0 or elem_dof_index == 5:
            return 0

        if elem_dof_index == 1 or elem_dof_index == 3:
            return - (6*(self.length - 2*z))/(self.length ** 3)

        if elem_dof_index == 2 or elem_dof_index == 4:
            return (6*z - 4*self.length)/(self.length**2)

        if elem_dof_index == 6 or elem_dof_index == 8:
            return (6*(self.length - 2*z))/(self.length**3)

        if elem_dof_index == 7 or elem_dof_index == 9:
            return - (2*(self.length-3*z))/(self.length**2)

    def get_deslocamentos(self, z):
        """Obtem os deslocamento axial e transversais nos planos zy
         e zx em um ponto qualquer z do elemento"""

        phi_i_u_i = []
        for n in range(10):
            phi_i_u_i.append(self.shape_functions(
                n, z)*self.deslocamentos_nodais[n])

        uo = phi_i_u_i[0] + phi_i_u_i[5]
        wy = phi_i_u_i[1] + phi_i_u_i[2] + phi_i_u_i[6] + phi_i_u_i[7]
        wx = phi_i_u_i[3] + phi_i_u_i[4] + phi_i_u_i[8] + phi_i_u_i[9]

        return uo, wy, wx

    def get_linear_forces_on_ends(self):
        """Obtem o vetor de forças nos nós da estrutura"""

        self.deslocamentos_nodais[:5] = self.i_node.deslocamentos[:5]
        self.deslocamentos_nodais[5:10] = self.j_node.deslocamentos[:5]
        self.linear_forces_on_ends = np.matmul(
            self.matriz_rigidez, self.deslocamentos_nodais)

    def get_deformacao_axial_curvaturas(self, z, nlg_expressions_only=False):
        """Obtem a deformacao axial e as curvaturas nos planos zy
         e zx em um ponto qualquer z do elemento"""

        phi_i_dz_u_i = []
        phi_i_dz2_u_i = []

        if nlg_expressions_only:
            for n in range(10):
                phi_i_dz_u_i.append(self.deriv_shape_functions(
                    n, z)*self.deslocamentos_nodais[n])

            dwy_dz = phi_i_dz_u_i[1] + phi_i_dz_u_i[2] + \
                phi_i_dz_u_i[6] + phi_i_dz_u_i[7]  # EXPRESSÃO A
            dwx_dz = phi_i_dz_u_i[3] + phi_i_dz_u_i[4] + \
                phi_i_dz_u_i[8] + phi_i_dz_u_i[9]  # EXPRESSÃO B

            return dwy_dz, dwx_dz

        for n in range(10):
            phi_i_dz_u_i.append(self.deriv_shape_functions(
                n, z)*self.deslocamentos_nodais[n])
            phi_i_dz2_u_i.append(self.second_deriv_shape_functions(
                n, z) * self.deslocamentos_nodais[n])

        duo_dz = phi_i_dz_u_i[0] + phi_i_dz_u_i[5]
        dwy_dz = phi_i_dz_u_i[1] + phi_i_dz_u_i[2] + \
            phi_i_dz_u_i[6] + phi_i_dz_u_i[7]  # EXPRESSÃO A
        dwx_dz = phi_i_dz_u_i[3] + phi_i_dz_u_i[4] + \
            phi_i_dz_u_i[8] + phi_i_dz_u_i[9]  # EXPRESSÃO B
        dwy_dz2 = phi_i_dz2_u_i[1] + phi_i_dz2_u_i[2] + \
            phi_i_dz2_u_i[6] + phi_i_dz2_u_i[7]
        dwx_dz2 = phi_i_dz2_u_i[3] + phi_i_dz2_u_i[4] + \
            phi_i_dz2_u_i[8] + phi_i_dz2_u_i[9]

        eo = duo_dz + 0.5 * (dwy_dz ** 2 + dwx_dz ** 2)   # DEFORMAÇÃO AXIAL
        chi_y = -dwy_dz2                                  # CURVATURA NO PLANO Z - Y
        chi_x = -dwx_dz2                                  # CURVATURA NO PLANO Z - X

        return eo, chi_y, chi_x

    def get_nonlinear_forces_vector(self):
        """Função que obtem o vetor de forças nao lineares no elemento"""

        self.deslocamentos_nodais[:5] = self.i_node.deslocamentos[:5]
        self.deslocamentos_nodais[5:10] = self.j_node.deslocamentos[:5]

        # Parametros necessarios para utiliar a integracao por gauss legendre
        [t, w] = np.polynomial.legendre.leggauss(3)
        j = 0.5 * (self.length)
        x = np.array([(j * i + j) for i in t])
        vetor_A = []
        vetor_B = []

        # Avaliação dos esforços internos em 3 pontos:

        k = np.sqrt(3/5)
        p1 = -k * self.length + 0.5*self.length
        p2 = 0.5 * self.length
        p3 = k * self.length + 0.5*self.length
        pontos_avaliacao = [p1, p2, p3]
        pontos_gauss_nd = []
        pontos_gauss_my = []
        pontos_gauss_mx = []

        for p in pontos_avaliacao:
            eo, chi_y, chi_x = self.get_deformacao_axial_curvaturas(p)
            nd_p, my_p, mx_p = self.secao_transversal.get_esforcos_internos(
                eo, chi_y, chi_x)
            pontos_gauss_nd.append(nd_p)
            pontos_gauss_my.append(my_p)
            pontos_gauss_mx.append(mx_p)

        pontos_gauss_nd = np.array([pontos_gauss_nd])
        pontos_gauss_my = np.array([pontos_gauss_my])
        pontos_gauss_mx = np.array([pontos_gauss_mx])

        for i in x:
            # Deformacao axial e curvaturas nos pontos necessarios para fazer a integração por gauss legendre
            # Expressões A e B avaliadas nos pontos para fazer a integração por gauss legendre
            dwy_dz, dwx_dz = self.get_deformacao_axial_curvaturas(
                i, nlg_expressions_only=True)
            # Esforços internos (Nd, My, Mz) nos pontos necessarios para fazer a integração por gauss legendre
            vetor_A.append(dwy_dz)
            vetor_B.append(dwx_dz)

        vetor_A = np.array(vetor_A)
        vetor_B = np.array(vetor_B)

        # Integração por gauss legendre para obtenção do vetor de forças nodais nao lineares
        # nos n graus de liberdade do elemento

        for n in range(10):
            if n == 0 or n == 5:
                self.non_linear_loads[n] = j * np.sum(
                    w * (pontos_gauss_nd * self.deriv_shape_functions(n, x)))

            if n == 1 or n == 2 or n == 6 or n == 7:
                expressao1 = j * \
                    np.sum(w * (- pontos_gauss_my *
                                self.second_deriv_shape_functions(n, x)))
                expressao2 = j * \
                    np.sum(w * (pontos_gauss_nd * vetor_A *
                                self.deriv_shape_functions(n, x)))
                self.non_linear_loads[n] = expressao1 + expressao2

            if n == 3 or n == 4 or n == 8 or n == 9:
                expressao1 = j * \
                    np.sum(w * (- pontos_gauss_mx *
                                self.second_deriv_shape_functions(n, x)))
                expressao2 = j * \
                    np.sum(w * (pontos_gauss_nd * vetor_B *
                                self.deriv_shape_functions(n, x)))
                self.non_linear_loads[n] = expressao1 + expressao2
    
    def check_ruptura(self):
        check_ruptura = self.secao_transversal.check_ruptura()
        if check_ruptura:
            return True
        else:
            return False


class Pilar:

    """Estrutura formada pelos n elementos finitos"""

    def __init__(self, node_list, bar_list):
        self.node_list = node_list
        self.bar_list = bar_list

        self.current_increment_ext_load_vect = None

        self.matriz_rigidez_inversa = None

        self.vetor_desloc_atual = np.zeros(5 * len(self.node_list))
        self.vetor_desloc_anterior = np.zeros(5 * len(self.node_list))

        self.vetor_desequilirio_forcas = np.zeros(5 * len(self.node_list))
        self.bound_cond_vector = None

        self.resultados_disponiveis = False

    def first_linear_solve(self, nr_increments):
        """Obtem a matriz inversa 'K' utilizada para determinar os deslocamentos em funcao de um
        vetor de cargas F, inicializa o pilar, transforma os carregamentos lineares em carregamentos
        nodais equivalentes, introduz as condições de contorno e resolve
         o sistema linear com o primeiro incremento de carga
        e """

        total_dof = len(self.node_list) * 5
        bound_cond_vector = []
        ext_load_vector = []
        glo_stiff_matx = np.zeros([total_dof, total_dof])

        # MONTAGEM DO VETOR DE CARGAS NODAIS E CONDIÇÕES DE CONTORNO
        for node in self.node_list:
            for n in range(5):
                bound_cond_vector.append(node.nodal_restrictions[n])
                ext_load_vector.append(node.nodal_loads[n])

        self.bound_cond_vector = np.array(bound_cond_vector)

        ext_load_vector = np.array(ext_load_vector)

        for bar in self.bar_list:
            # INDICES DOS DOF DOS NOS DA BARRA NOS VETORES GLOBAIS
            i_index = 5 * bar.i_node.index
            j_index = 5 * bar.j_node.index

            # MONTAGEM DA MATRIZ "K" DE RIGIDEZ GLOBAL
            bar.get_elem_stiff_mat()
            glo_stiff_matx[i_index:i_index + 5,
                           i_index:i_index + 5] += bar.matriz_rigidez[:5, :5]
            glo_stiff_matx[i_index:i_index + 5,
                           j_index:j_index + 5] += bar.matriz_rigidez[:5, 5:10]
            glo_stiff_matx[j_index:j_index + 5,
                           i_index:i_index + 5] += bar.matriz_rigidez[5:10, :5]
            glo_stiff_matx[j_index:j_index + 5, j_index:j_index +
                           5] += bar.matriz_rigidez[5:10, 5:10]

            # ADIÇÃO DAS CARGAS DISTRIBUIDAS NAS BARRAS EQUIVALENTES NOS NÓS
            bar.get_equiv_nodal_loads()
            ext_load_vector[i_index:i_index + 5] += bar.equiv_nodal_loads[:5]
            ext_load_vector[j_index:j_index + 5] += bar.equiv_nodal_loads[5:10]

        # APLCAÇÃO DAS CONDIÇÕES DE CONTORNO NA MATRIZ DE RIGIDEZ "K"
        for n in range(total_dof):
            for m in range(total_dof):
                if self.bound_cond_vector[n] == 0:
                    if n == m:
                        glo_stiff_matx[n][m] = 1
                    else:
                        glo_stiff_matx[n][m] = 0
                        glo_stiff_matx[m][n] = 0

        # MATRIZ 'K' INVERSA E VETOR DE CARGAS EXTERNAS

        self.matriz_rigidez_inversa = np.linalg.inv(glo_stiff_matx)
        ext_load_vector = ext_load_vector * self.bound_cond_vector
        self.load_increment = np.copy(ext_load_vector)/nr_increments

        # Resolve o sistema linear com o primeiro incremento das forças nodais externas
        self.current_increment_ext_load_vect = np.copy(self.load_increment)
        self.vetor_desloc_anterior = np.array(
            [i for i in self.vetor_desloc_atual])

        # APLICAÇÃO DAS CONDIÇÕES DE CONTORNO NO VETOR DE FORÇAS
        load_vector_bc = self.current_increment_ext_load_vect * self.bound_cond_vector
        self.vetor_desloc_atual = np.matmul(
            self.matriz_rigidez_inversa, load_vector_bc)

        # ATUALIZA OS DESLOCAMENTOS NOS NÓS
        for node in self.node_list:
            node.deslocamentos[:5] = self.vetor_desloc_atual[5 *
                                                             node.index:5 * node.index + 5]

    def extrapol_deslocamento(self, increment_atu):

        if increment_atu == 1:
            self.vetor_un = np.array([i for i in self.vetor_desloc_atual])

        if increment_atu == 2:
            self.vetor_un_ant = np.array([i for i in self.vetor_un])
            self.vetor_un = np.array([i for i in self.vetor_desloc_atual])
            self.vetor_desloc_atual = (2 * self.vetor_un) + self.vetor_un_ant

        if increment_atu == 3:
            self.vetor_un_ant_ant = np.array([i for i in self.vetor_un_ant])
            self.vetor_un_ant = np.array([i for i in self.vetor_un])
            self.vetor_un = np.array([i for i in self.vetor_desloc_atual])
            self.vetor_desloc_atual = (
                3 * self.vetor_un) - (3 * self.vetor_un_ant) + self.vetor_un_ant_ant

        if increment_atu >= 4:
            self.vetor_un_ant_ant_ant = np.array(
                [i for i in self.vetor_un_ant_ant])
            self.vetor_un_ant_ant = np.array([i for i in self.vetor_un_ant])
            self.vetor_un_ant = np.array([i for i in self.vetor_un])
            self.vetor_un = np.array([i for i in self.vetor_desloc_atual])
            self.vetor_desloc_atual = (4 * self.vetor_un) - (6 * self.vetor_un_ant) + (
                4 * self.vetor_un_ant_ant) - self.vetor_un_ant_ant_ant

    def j_linear_solve(self):
        """Resolve o sistema linear com a diferença entra as forças nodais
        externas e forças nodais nao lineares"""

        self.vetor_desloc_anterior = np.array(
            [i for i in self.vetor_desloc_atual])

        # APLICAÇÃO DAS CONDIÇÕES DE CONTORNO NO VETOR DE FORÇAS
        load_vector_bc = self.vetor_desequilirio_forcas * self.bound_cond_vector

        self.vetor_desloc_atual += np.matmul(
            self.matriz_rigidez_inversa, load_vector_bc)

        # ATUALIZA OS DESLOCAMENTOS NODAIS
        for node in self.node_list:
            node.deslocamentos[:5] = self.vetor_desloc_atual[5 *
                                                             node.index:5 * node.index + 5]

    def convergencia_incremento(self, tolerancia_f, tolerancia_u, n_iter):
        """Verifica a convergencia dos esforços e dos deslocamentos para um dado
        incremento de carga"""

        a = np.linalg.norm(self.vetor_desequilirio_forcas)
        b = np.linalg.norm(self.current_increment_ext_load_vect)
        c = np.linalg.norm(self.vetor_desloc_atual -
                           self.vetor_desloc_anterior)
        d = np.linalg.norm(self.vetor_desloc_atual)

        print('convergencia de forcas: ', a/b)
        print('convergencia de deslocamentos: ', c/d)

        if (a/b <= tolerancia_f) and (c/d <= tolerancia_u) and n_iter > 0:
            return True

        return False

    def non_linear_solve(self, tolerancia_f, tolerancia_u, n_increments, max_iter):
        """Solver não linear do pilar, realizando os incrementos de cargas
        e verificando se existe convergencia dos esforços e deslocamentos"""

        iteracoes = 0
        start_time = time.time()

        self.first_linear_solve(n_increments)
        current_increment = 1
        n_iter = 0

        while True:

            # Calcula o vetor de forças nao lineares
            global_non_linear_load_vector = np.zeros(5 * len(self.node_list))
            for bar in self.bar_list:
                bar.get_nonlinear_forces_vector()
                i_index = 5 * bar.i_node.index
                j_index = 5 * bar.j_node.index
                global_non_linear_load_vector[i_index:i_index +
                                              5] += bar.non_linear_loads[:5]
                global_non_linear_load_vector[j_index:j_index +
                                              5] += bar.non_linear_loads[5:10]

            global_non_linear_load_vector = np.array(
                global_non_linear_load_vector)

            global_non_linear_load_vector = self.bound_cond_vector * \
                global_non_linear_load_vector

            # Desequilibrio entre vetor de forças nao lineares Fj e vetor de cargas externas F
            # no incremento atual
            self.vetor_desequilirio_forcas = np.subtract(
                self.current_increment_ext_load_vect, global_non_linear_load_vector)

            # verificação da convergencia
            print('incremento atual: ', current_increment,
                  ' iteração atual: ', n_iter)
            if self.convergencia_incremento(tolerancia_f, tolerancia_u, n_iter):
                if current_increment == n_increments:
                    end_time = time.time()
                    print('tempo de execução: ', end_time - start_time)
                    return True

                self.current_increment_ext_load_vect += self.load_increment
                self.vetor_desequilirio_forcas = self.current_increment_ext_load_vect - \
                    global_non_linear_load_vector
                self.extrapol_deslocamento(current_increment)

                current_increment += 1
                n_iter = 0

            elif n_iter > max_iter:
                return False

            n_iter += 1
            self.j_linear_solve()
            iteracoes += 1

    def get_resultados(self, secao, tol_ln, pontos_diag):
        uy = np.zeros([len(self.node_list) + len(self.bar_list)])
        ux = np.zeros([len(self.node_list) + len(self.bar_list)])
        uz = np.zeros([len(self.node_list) + len(self.bar_list)])
        nd = np.zeros([len(self.node_list) + len(self.bar_list)])
        my = np.zeros([len(self.node_list) + len(self.bar_list)])
        mx = np.zeros([len(self.node_list) + len(self.bar_list)])
        envoltorias = []
        solicitacoes = []
        altura_corresp = []
        pontos_comprimento = np.zeros(
            [len(self.node_list) + len(self.bar_list)])

        for bar in self.bar_list:
            index = bar.i_node.index*2
            pontos_z_global = [bar.i_node.z_pos,
                               bar.i_node.z_pos + bar.length/2, bar.j_node.z_pos]
            pontos_z_local = [0, 0.5*bar.length, bar.length]
            bar.deslocamentos_nodais[:5] = bar.i_node.deslocamentos[:5]
            bar.deslocamentos_nodais[5:10] = bar.j_node.deslocamentos[:5]

            for i, p in enumerate(pontos_z_local):
                uo, wy, wx = bar.get_deslocamentos(p)
                eo, chi_y, chi_z = bar.get_deformacao_axial_curvaturas(p)
                nd_p, my_p, mx_p = bar.secao_transversal.get_esforcos_internos(
                    eo, chi_y, chi_z, fluen=True)

                if index == 0:
                    if i == 2:
                        nd[index + i] -= nd_p/2
                        mx[index + i] += my_p/2
                        my[index + i] -= mx_p/2
                        uz[index + i] += uo/2
                        uy[index + i] += wy/2
                        ux[index + i] += wx/2
                        pontos_comprimento[index + i] += pontos_z_global[i]/2
                    else:
                        nd[index + i] -= nd_p
                        mx[index + i] += my_p
                        my[index + i] -= mx_p
                        uz[index + i] += uo
                        uy[index + i] += wy
                        ux[index + i] += wx
                        pontos_comprimento[index + i] += pontos_z_global[i]

                elif index == (len(self.node_list) - 2) * 2:
                    if i == 0:
                        nd[index + i] -= nd_p/2
                        mx[index + i] += my_p/2
                        my[index + i] -= mx_p/2
                        uz[index + i] += uo/2
                        uy[index + i] += wy/2
                        ux[index + i] += wx/2
                        pontos_comprimento[index + i] += pontos_z_global[i] / 2
                    else:
                        nd[index + i] -= nd_p
                        mx[index + i] += my_p
                        my[index + i] -= mx_p
                        uz[index + i] += uo
                        uy[index + i] += wy
                        ux[index + i] += wx
                        pontos_comprimento[index + i] += pontos_z_global[i]

                else:
                    if i % 2 == 0:
                        nd[index + i] -= nd_p/2
                        mx[index + i] += my_p/2
                        my[index + i] -= mx_p/2
                        uz[index + i] += uo/2
                        uy[index + i] += wy/2
                        ux[index + i] += wx/2
                        pontos_comprimento[index + i] += pontos_z_global[i] / 2
                    else:
                        nd[index + i] -= nd_p
                        mx[index + i] += my_p
                        my[index + i] -= mx_p
                        uz[index + i] += uo
                        uy[index + i] += wy
                        ux[index + i] += wx
                        pontos_comprimento[index + i] += pontos_z_global[i]

        ind_ndmax = np.argmax(nd)
        ind_ndmin = np.argmin(nd)
        ind_mxmax = np.argmax(mx)
        ind_mxmin = np.argmin(mx)
        ind_mymax = np.argmax(my)
        ind_mymin = np.argmin(my)

        pontos_aval = [ind_ndmax, ind_ndmin,
                       ind_mxmax, ind_mxmin, ind_mymax, ind_mymin]
        for p in pontos_aval:
            secao.get_diagrama_n_mx_my(nd[p], tol_ln, pontos_diag)
            envoltorias.append(
                np.array([i for i in secao.envoltoria_momentos]))
            solicitacoes.append(np.array([mx[p], my[p]]))
            altura_corresp.append(pontos_comprimento[p])

        ruptura = False
        rupturaNd = False
        for bar in self.bar_list:
            ruptura = bar.check_ruptura()
            if ruptura:
                break
            if bar.secao_transversal.rupturaNd:
                rupturaNd = True

        return pontos_comprimento, uy, ux, uz, nd, my, mx, envoltorias, solicitacoes, altura_corresp, ruptura, rupturaNd
