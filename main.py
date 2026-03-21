import pandas as pd
import random
import math

class NeuronalesNetz:
    def __init__(self, file):
        if not isinstance(file, str):
            raise TypeError("You must provide teh directory")
        # get Data
        data = pd.read_excel(file)
        rows = ["Energie (kJ)", "Gesätt. Fetts. (g)", "Zucker (g)", "Salz (g)", "Ballastst. (g)", "Eiweiß (g)", "Obst/Gem. (%)"]
        self.input_val = [data[rows].iloc[i].tolist() for i in range(len(data))]
        #Struktere
        self.input_num = 7
        self.output_val = 5
        self.Hidden_Layer = 1
        self.Knoten_pL = self.input_num + 2
        #Gewichtungen
        self.base = self.create_base_vector([random.choice([-2,-1,-0,1,2]) for i in range(self.input_num+self.Knoten_pL*self.Hidden_Layer+self.output_val)])
        self.weight = self.create_matrix([random.choice([-2,-1,-0,1,2]) for i in range((self.input_num*self.Knoten_pL)+((self.Knoten_pL*self.Knoten_pL)*self.Hidden_Layer)+(self.output_val*self.Knoten_pL))])

    #vector/matrix definition
    def create_matrix(self, val_w):
        matrix_ges = []
        matrix_layer = []
        #matrix_Knoten = []
        ak_pos = 0

        # entry-layer
        for _ in range(self.Knoten_pL):
            matrix_Knoten = val_w[ak_pos : ak_pos+self.input_num]
            matrix_layer.append(matrix_Knoten)
            ak_pos += self.input_num
        matrix_ges.append(matrix_layer)
        matrix_layer = []

        # n: hidden-layer
        """if self.Hidden_Layer >1:
            for _ in range(self.Hidden_Layer):
                for _ in range(self.Knoten_pL):
                    matrix_Knoten = val_w[ak_pos : ak_pos + self.Knoten_pL]
                    matrix_layer.append(matrix_Knoten)
                    ak_pos += self.Knoten_pL
                matrix_ges.append(matrix_layer)
                matrix_layer = []"""

        # output-layer
        for _ in range(self.output_val):
            matrix_Knoten = val_w[ak_pos : ak_pos + self.Knoten_pL]
            matrix_layer.append(matrix_Knoten)
        matrix_ges.append(matrix_layer)
        return matrix_ges
    def create_base_vector(self, base_ges):
        base_ges_format = []
        basis_Layer =[]
        akt_pos = 0

        for _ in range(self.Hidden_Layer):
            for _ in range(self.Knoten_pL):
                basis_Layer.append(base_ges[akt_pos])
                akt_pos += 1
            base_ges_format.append(basis_Layer)
            basis_Layer = []
        for _ in range(self.output_val):
            basis_Layer.append(base_ges[akt_pos])
            akt_pos += 1
        base_ges_format.append(basis_Layer)
        return base_ges_format

    #funktionen
    def Relu(self, i):
        if i <= 0:
            return 0
        else:
            return i
    def Softmax(self, i):
        max_val = max(i)
        e_werte = [math.exp(x - max_val) for x in i]
        nenner_summe = sum(e_werte)
        erg = [k / nenner_summe for k in e_werte]

        return erg

    #funktionen des Neuronalen Netzes
    def forwardpropagation(self):
        # i = Tuplet aus 8 Werten
        ges_output = []
        for val in self.input_val:
            eingabe_werte = [k for k in val]
            aktuelle_Layer = []
            equation = 0
            # für Knoten vor der Outputschicht Relu-Funktion

            # input & hidden
            for l in range(len(self.weight)):
                for j in range(len(self.weight[l])): #für jeden Knoten einmal
                    for k in range(len(self.weight[l][j])): #welche werte leigen aus letzter Schicht vor?
                        equation += eingabe_werte[k] * self.weight[l][j][k]
                    aktuelle_Layer.append(self.Relu(equation + self.base[l][j]))
                eingabe_werte = aktuelle_Layer
                aktuelle_Layer = []
            #output
            final_values = []
            for j in range(self.output_val):  # für jeden Knoten einmal
                for k in range(len(eingabe_werte)):
                    equation += eingabe_werte[k] * self.weight[-1][j][k]
                final_values.append(equation + self.base[-1][j])
            ges_output.append(self.Softmax(final_values))
            max_index = [ges_output[k].index(max(ges_output[k])) for k in range(len(ges_output))]
            return_output = [self.Nutriscore(k) for k in max_index]
        return return_output
    def Nutriscore(self, k):
        match(k):
            case 0:
                return "A"
            case 1:
                return "B"
            case 2:
                return "C"
            case 3:
                return "D"
            case 4:
                return "F"


neu =NeuronalesNetz("Trainingsdaten.xlsx")
print(neu.forwardpropagation())