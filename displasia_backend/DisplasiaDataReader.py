import pandas as pd
import numpy as np
class dataReader:
    def __init__(self, FeaturesTypesUsed = 'all'):
        self.data = {}
        self.colNames = []
        self.engNames = np.loadtxt('Data/Atributes/AtributesEnglish.txt',dtype=str)
        self.class_names = ['healthy','mild','moderate','severe']
        for estado in ['healthy','mild','moderate','severe']:
            self.data[estado] = pd.read_csv('Data/Displasia/caracteristicas-' + estado + '.csv',header = 0)
        
        if(FeaturesTypesUsed == 'all'):
            self.SelectUsedColumnsAll()
        elif(FeaturesTypesUsed == 'morphological'):
            self.SelectUsedColumnsMorphological()
        elif(FeaturesTypesUsed == 'nonmorphological'):
            self.SelectUsedColumnsNonMorphological()
        else:
            raise 'tipo nao existe'

        for estado in self.data.keys():
            self.data[estado] = self.data[estado].astype(float)

    
    def SelectUsedColumnsAll(self):
        self.ColumnsUsedNames = np.loadtxt('Data/Atributes/AtributesUsed.txt',dtype=str)
        for estado in self.data.keys():
            self.data[estado] = self.data[estado][self.ColumnsUsedNames]
        
    def SelectUsedColumnsMorphological(self):
        self.ColumnsUsedNames = np.loadtxt('Data/Atributes/MorphologicalAtributesUsed.txt',dtype=str)
        for estado in self.data.keys():
            self.data[estado] = self.data[estado][self.ColumnsUsedNames]
    
    def SelectUsedColumnsNonMorphological(self):
        self.ColumnsUsedNames = np.loadtxt('Data/Atributes/NonMorphologicalAtributesUsed.txt',dtype=str)
        for estado in self.data.keys():
            self.data[estado] = self.data[estado][self.ColumnsUsedNames]

    def GetClasses(self,classNamesList):
        #return a array with all the data from the classes on the classesList parameter
        classList = []
        for i,className in enumerate(classNamesList):
            x = self.data[className].copy()
            x['Target'] = [i]*x.shape[0]
            classList.append(x)
        
        return pd.concat(classList,ignore_index=True)

    def GetData(self):
        dataBase = self.GetClasses(['healthy','mild','moderate','severe'])
        data = dataBase.drop(labels='Target',axis = 1)
        target = dataBase['Target']
        return data, target
