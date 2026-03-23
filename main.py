import pandas as pd
import random
import math

class NeuronalesNetz:
    def __init__(self, file, input_num, output_num, Hidden_num, Knoten_num):
        if not isinstance(file, str):
            raise TypeError("You must provide teh directory")
        # get Data
        data = pd.read_excel(file)
        rows = ["Energie (kJ)", "Gesätt. Fetts. (g)", "Zucker (g)", "Salz (g)", "Ballastst. (g)", "Eiweiß (g)", "Obst/Gem. (%)"]
        self.input_val = [data[rows].iloc[i].tolist() for i in range(len(data))]
        erwartung = [data[["Score"]].iloc[i].tolist() for i in range(len(data))]
        self.erwartung_val = [self.reverse_nutriscore(i) for i in erwartung]
        print(self.erwartung_val)
        #Struktere
        self.structure = [input_num, output_num, Hidden_num, Knoten_num]
        #Gewichtungen
        self.base = self.create_base_vector([random.choice([-2,-1,-0,1,2]) for i in range(self.structure[0]+self.structure[3]*self.structure[2]+self.structure[1])])
        self.weight = self.create_matrix([random.choice([-2,-1,-0,1,2]) for i in range((self.structure[0]*self.structure[3])+((self.structure[3]*self.structure[3])*self.structure[2])+(self.structure[1]*self.structure[3]))])

    #vector/matrix definition
    def create_matrix(self, val_w):
        matrix_ges = []
        matrix_layer = []
        #matrix_Knoten = []
        ak_pos = 0

        # entry-layer
        for _ in range(self.structure[3]):
            matrix_Knoten = val_w[ak_pos : ak_pos+self.structure[0]]
            matrix_layer.append(matrix_Knoten)
            ak_pos += self.structure[0]
        matrix_ges.append(matrix_layer)
        matrix_layer = []

        # n: hidden-layer
        if self.structure[2] >1:
            for _ in range(self.structure[2]):
                for _ in range(self.structure[3]):
                    matrix_Knoten = val_w[ak_pos : ak_pos + self.structure[3]]
                    matrix_layer.append(matrix_Knoten)
                    ak_pos += self.structure[3]
                matrix_ges.append(matrix_layer)
                matrix_layer = []

        # output-layer
        for _ in range(self.structure[1]):
            matrix_Knoten = val_w[ak_pos : ak_pos + self.structure[3]]
            matrix_layer.append(matrix_Knoten)
        matrix_ges.append(matrix_layer)
        return matrix_ges
    def create_base_vector(self, base_ges):
        base_ges_format = []
        basis_Layer =[]
        akt_pos = 0

        for _ in range(self.structure[2]):
            for _ in range(self.structure[3]):
                basis_Layer.append(base_ges[akt_pos])
                akt_pos += 1
            base_ges_format.append(basis_Layer)
            basis_Layer = []

        if self.structure[2] > 1:
            for _ in range(self.structure[2]):
                for _ in range(self.structure[3]):
                    basis_Layer.append(base_ges[akt_pos])
                    akt_pos += 1
                base_ges_format.append(basis_Layer)
                basis_Layer = []

        for _ in range(self.structure[1]):
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

    #output evaluateion
    def nutriscore(self, k):
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
    def reverse_nutriscore(self,k):
        match(k):
            case ["A"]:
                return 0
            case ["B"]:
                return 1
            case ["C"]:
                return 2
            case ["D"]:
                return 3
            case ["E"]:
                return 4

    #funktionen des Neuronalen Netzes
    def forwardpropagation(self, training= False):
        # i = Tuplet aus 8 Werten
        ges_output = []
        for val in self.input_val:
            eingabe_werte = [k for k in val]
            aktuelle_Layer = []
            alle_Knoten = []
            equation = 0
            # für Knoten vor der Outputschicht Relu-Funktion

            # input & hidden
            for l in range(len(self.weight) - self.structure[1]):
                for j in range(len(self.weight[l])): #für jeden Knoten einmal
                    for k in range(len(self.weight[l][j])): #welche werte leigen aus letzter Schicht vor?
                        equation += eingabe_werte[k] * self.weight[l][j][k]
                    aktuelle_Layer.append(self.Relu(equation + self.base[l][j]))
                eingabe_werte = aktuelle_Layer
                aktuelle_Layer = []
            #output
            final_values = []
            for j in range(self.structure[1]):  # für jeden Knoten einmal
                for k in range(len(eingabe_werte)):
                    equation += eingabe_werte[k] * self.weight[-1][j][k]
                final_values.append(equation + self.base[-1][j])
            ges_output.append(self.Softmax(final_values))

            #output protokoll
        max_index_pos = [ges_output[k].index(max(ges_output[k])) for k in range(len(ges_output))]
        return_output = [self.nutriscore(k) for k in max_index_pos]
        if training:
            return return_output, eingabe_werte
        return return_output
    def Fehlerrate(self, results):
        error = []
        for i in range(len(results)):
            calcualtion = (1.0 - self.erwartung_val[i])**2
            error.append(calcualtion)
        return error
    def backpropagation(self):
        erg, Knoten_ges = self.forwardpropagation(True)
        Knotne_für_rechnene = [self.input_val, Knoten_ges]
        print(Knotne_für_rechnene)
        for i in range(len(self.weight)):
            for k in range(len(self.weight[i])):
                for j in range(len(self.weight[i][k])):
                    print(self.Fehlerrate(erg))
                    self.weight[i][j][k] += 0.1 * (self.Fehlerrate(erg)) * self.input_val


neu =NeuronalesNetz("Trainingsdaten.xlsx",7,5,1,9)
print(neu.forwardpropagation())
print(neu.backpropagation())