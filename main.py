import pandas as pd
import random
import math

class NeuronalesNetz:
    def __init__(self, file, input_num, output_num, Hidden_num, Knoten_num, manuell=False, base_val = None, weight_val = None):
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
        if not manuell:
            self.base = self.create_base_vector([random.choice([-2,-1,-0,1,2]) for i in range(self.structure[0]+self.structure[3]*self.structure[2]+self.structure[1])])
            self.weight = self.create_matrix([random.choice([-2,-1,-0,1,2]) for i in range((self.structure[0]*self.structure[3])+((self.structure[3]*self.structure[3])*self.structure[2])+(self.structure[1]*self.structure[3]))])
            print(f"base: {self.base}")
            print(f"weight: {self.weight}")
        else:
            print(f"base: {base_val}")
            print(f"weight: {weight_val}")
            self.base = base_val
            self.weight = weight_val
            
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
                return "E"
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
<<<<<<< HEAD
        # print(ges_output)
        return return_output, ges_output
    
    def Fehlerrate(self, results, ges_output):
        error = {}
        # erwartung_val = [self.reverse_nutriscore(i) for i in self.erwartung_val]
        # print(self.erwartung_val)
        # print(erwartung_val)
        for n,i in enumerate(results):
            expected = self.erwartung_val[n]
            # print(expected)
            list_item = self.reverse_nutriscore([i])
            all_values = ges_output[n]
            calcualtion = (1.0 - all_values[expected])**2
            error[n] = [list_item, calcualtion]
        # print(error)
        return error
    def backpropagation(self, epoch=100):
        for epochs in range(epoch):
            inefficient = False
            forwardp = self.forwardpropagation()
            erg, ges_output = forwardp[0], forwardp[1]
            all_error = [self.Fehlerrate(erg, ges_output)[i][1] for i in range(self.Fehlerrate(erg, ges_output).__len__())]
            print(all_error)
            complete_error = [True for i in all_error if i > 0.01]
            if False in complete_error:
                for sample_idx, input_sample in enumerate(self.input_val):
                    for i in range(len(self.weight)):
                        for j in range(len(self.weight[i])):
                            for k in range(len(self.weight[i][j])):
                                error = self.Fehlerrate(erg[sample_idx], ges_output[sample_idx])[sample_idx][1]
                                if self.weight[i][j][k] > 0:
                                    self.weight[i][j][k] -= 0.1 * error * input_sample[k]
                                else:
                                    self.weight[i][j][k] += 0.1 * error * input_sample[k]

        return self.weight , self.base
=======
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
>>>>>>> 23f56f02ce8db69057420ee9048262b1c5534deb


neu =NeuronalesNetz("Trainingsdaten.xlsx",7,5,1,9)
print(neu.forwardpropagation())
<<<<<<< HEAD
frorw = neu.forwardpropagation()
print(neu.Fehlerrate(frorw[0], frorw[1]))
next_back = neu.backpropagation()
print(next_back)
next = NeuronalesNetz("Trainingsdaten.xlsx",7,5,1,9, manuell=True, base_val=next_back[1], weight_val=next_back[0])
print(next.forwardpropagation()[0])
=======
print(neu.backpropagation())
>>>>>>> 23f56f02ce8db69057420ee9048262b1c5534deb
