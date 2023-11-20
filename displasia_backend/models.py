from typing import List, Optional
from enum import Enum
from sqlmodel import Field, Relationship, SQLModel
from pandas import DataFrame
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
#from sklearn.neural_network import MLPClassifier

class Classe(str, Enum):
    HEALTHY = 'healthy'
    MILD = 'mild'
    MODERATE = 'moderate'
    SEVERE = 'severe'
    
class ClasseBr(str, Enum):
    SAUDAVEL = 'Saudavel'
    LEVE = 'Leve'
    MODERADA = 'Moderada'
    SEVERA = 'Severa'

class Confianca(str, Enum):
    CONFIAVEL = 'Confiavel'
    INCERTA = 'Incerta'
    INCONCLUSIVA = 'Inconclusiva'

class Amostra(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    imagem: str
    classe: Classe
    atributos: List["Atributo"] = Relationship(back_populates="amostra")

class Atributo(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    amostra_id: Optional[int] = Field(default=None, foreign_key="amostra.id")
    nome: str
    valor: float
    amostra: Amostra = Relationship(back_populates="atributos")

class LocalShap(BaseModel):
    name: str = "Shap Local"
    path: Optional[str]
    shap_values: Optional[List[float]]

class LocalAnchor(BaseModel):
    name: str = "Anchor Local"
    path: Optional[str]
    healthy: Optional[str]
    mild: Optional[str]
    moderate: Optional[str]
    severe: Optional[str]
    predicted_class: Optional[Classe]

class LocalExp(BaseModel):
    shap: LocalShap
    anchor: LocalAnchor

class GlobalShap(BaseModel):
    path: str = "displasia_backend/static/shap.jpg"

class GlobalCorr(BaseModel):
    path: str = "displasia_backend/static/corr.jpg"

class GlobalData(BaseModel):
    class_names: List[str] = [Classe.HEALTHY.value,Classe.MILD.value,Classe.MODERATE.value,Classe.SEVERE.value]
    global_shap: GlobalShap = GlobalShap()
    global_corr: GlobalCorr = GlobalCorr()
    data:        Optional[DataFrame]
    target: Optional[DataFrame]
    train_data:  Optional[DataFrame]
    test_data:  Optional[DataFrame]
    train_target:  Optional[DataFrame]
    test_target:  Optional[DataFrame]
    scaler: Optional[StandardScaler]
    scaled_train_data:  Optional[DataFrame]
    scaled_test_data:  Optional[DataFrame]
    class Config:
        arbitrary_types_allowed = True
    #model: Optional[MLPClassifier]

class Imagem(BaseModel):
    URL: str

class AmostraResponse(BaseModel):
    id: int
    imagens: List[Imagem]
    classificacao: ClasseBr
    explicacaoPrincipal: LocalExp
    amostrasSimilares: List[int]
    confianca: Confianca

#static class
class Names(BaseModel):
    attributes_used: List[str] = [
    "Media_Areas",
    "Media_Extents",
    "Media_Perimeters",
    "Media_ConvexAreas",
    "Media_Soliditys",
    "Media_Eccentricitys",
    "Media_MajorAxisLengths",
    "Media_MinorAxisLengths",
    "Media_EquivDiameters",
    "Media_ValorPixelMinimoIr",
    "Media_ValorPixelMaximoIr",
    "Media_ValorPixelMediaIr",
    "Media_ValorPixelMedianaIr",
    "Media_ValorPixelDesvioPadraoIr",
    "Media_ValorPixelMinimoIg",
    "Media_ValorPixelMaximoIg",
    "Media_ValorPixelMediaIg",
    "Media_ValorPixelMedianaIg",
    "Media_ValorPixelDesvioPadraoIg",
    "Media_ValorPixelMinimoIb",
    "Media_ValorPixelMaximoIb",
    "Media_ValorPixelMediaIb",
    "Media_ValorPixelMedianaIb",
    "Media_ValorPixelDesvioPadraoIb",
    "Media_ValorPixelMinimoIc",
    "Media_ValorPixelMaximoIc",
    "Media_ValorPixelMediaIc",
    "Media_ValorPixelMedianaIc",
    "Media_ValorPixelDesvioPadraoIc",
    "Mediana_Extents",
    "Mediana_Perimeters",
    "Mediana_ConvexAreas",
    "Mediana_Soliditys",
    "Mediana_Eccentricitys",
    "Mediana_MajorAxisLengths",
    "Mediana_MinorAxisLengths",
    "Mediana_EquivDiameters",
    "Mediana_ValorPixelMinimoIr",
    "Mediana_ValorPixelMaximoIr",
    "Mediana_ValorPixelMediaIr",
    "Mediana_ValorPixelMedianaIr",
    "Mediana_ValorPixelDesvioPadraoIr",
    "Mediana_ValorPixelMinimoIg",
    "Mediana_ValorPixelMaximoIg",
    "Mediana_ValorPixelMediaIg",
    "Mediana_ValorPixelMedianaIg",
    "Mediana_ValorPixelDesvioPadraoIg",
    "Mediana_ValorPixelMinimoIb",
    "Mediana_ValorPixelMaximoIb",
    "Mediana_ValorPixelMediaIb",
    "Mediana_ValorPixelMedianaIb",
    "Mediana_ValorPixelDesvioPadraoIb",
    "Mediana_ValorPixelMinimoIc",
    "Mediana_ValorPixelMaximoIc",
    "Mediana_ValorPixelMediaIc",
    "Mediana_ValorPixelMedianaIc",
    "Mediana_ValorPixelDesvioPadraoIc",
    "Desvio_Areas",
    "Desvio_Extents",
    "Desvio_Perimeters",
    "Desvio_ConvexAreas",
    "Desvio_Soliditys",
    "Desvio_Eccentricitys",
    "Desvio_MajorAxisLengths",
    "Desvio_MinorAxisLengths",
    "Desvio_EquivDiameters",
    "Desvio_ValorPixelMinimoIr",
    "Desvio_ValorPixelMaximoIr",
    "Desvio_ValorPixelMediaIr",
    "Desvio_ValorPixelMedianaIr",
    "Desvio_ValorPixelDesvioPadraoIr",
    "Desvio_ValorPixelMinimoIg",
    "Desvio_ValorPixelMaximoIg",
    "Desvio_ValorPixelMediaIg",
    "Desvio_ValorPixelMedianaIg",
    "Desvio_ValorPixelDesvioPadraoIg",
    "Desvio_ValorPixelMinimoIb",
    "Desvio_ValorPixelMaximoIb",
    "Desvio_ValorPixelMediaIb",
    "Desvio_ValorPixelMedianaIb",
    "Desvio_ValorPixelDesvioPadraoIb",
    "Desvio_ValorPixelMinimoIc",
    "Desvio_ValorPixelMaximoIc",
    "Desvio_ValorPixelMediaIc",
    "Desvio_ValorPixelMedianaIc",
    "Desvio_ValorPixelDesvioPadraoIc",
    "Moda_Areas",
    "Moda_Extents",
    "Moda_Perimeters",
    "Moda_ConvexAreas",
    "Moda_Eccentricitys",
    "Moda_MajorAxisLengths",
    "Moda_MinorAxisLengths",
    "Moda_EquivDiameters",
    "Moda_ValorPixelMinimoIr",
    "Moda_ValorPixelMaximoIr",
    "Moda_ValorPixelMediaIr",
    "Moda_ValorPixelMedianaIr",
    "Moda_ValorPixelDesvioPadraoIr",
    "Moda_ValorPixelMinimoIg",
    "Moda_ValorPixelMaximoIg",
    "Moda_ValorPixelMediaIg",
    "Moda_ValorPixelMedianaIg",
    "Moda_ValorPixelDesvioPadraoIg",
    "Moda_ValorPixelMinimoIb",
    "Moda_ValorPixelMaximoIb",
    "Moda_ValorPixelMediaIb",
    "Moda_ValorPixelMedianaIb",
    "Moda_ValorPixelDesvioPadraoIb",
    "Moda_ValorPixelMinimoIc",
    "Moda_ValorPixelMaximoIc",
    "Moda_ValorPixelMediaIc",
    "Moda_ValorPixelMedianaIc",
    "Moda_ValorPixelDesvioPadraoIc"
    ]
    eng_names: List[str] = [
        'Avr_Areas',
        'Avr_Extents',
        'Avr_Perimeters',
        'Avr_ConvexAreas',
        'Avr_Soliditys',
        'Avr_Eccentricitys',
        'Avr_MajorAxisLengths',
        'Avr_MinorAxisLengths',
        'Avr_EquivDiameters',
        'Avr_Min_r',
        'Avr_Max_r',
        'Avr_Avr_r',
        'Avr_Median_r',
        'Avr_StdDev_r',
        'Avr_Min_g',
        'Avr_Max_g',
        'Avr_Avr_g',
        'Avr_Median_g',
        'Avr_StdDev_g',
        'Avr_Min_b',
        'Avr_Max_b',
        'Avr_Avr_b',
        'Avr_Median_b',
        'Avr_StdDev_b',
        'Avr_Min_gray',
        'Avr_Max_gray',
        'Avr_Avr_gray',
        'Avr_Median_gray',
        'Avr_StdDev_gray',
        'Median_Extents',
        'Median_Perimeters',
        'Median_ConvexAreas',
        'Median_Soliditys',
        'Median_Eccentricitys',
        'Median_MajorAxisLengths',
        'Median_MinorAxisLengths',
        'Median_EquivDiameters',
        'Median_Min_r',
        'Median_Max_r',
        'Median_Avr_r',
        'Median_Median_r',
        'Median_StdDev_r',
        'Median_Min_g',
        'Median_Max_g',
        'Median_Avr_g',
        'Median_Median_g',
        'Median_StdDev_g',
        'Median_Min_b',
        'Median_Max_b',
        'Median_Avr_b',
        'Median_Median_b',
        'Median_StdDev_b',
        'Median_Min_gray',
        'Median_Max_gray',
        'Median_Avr_gray',
        'Median_Median_gray',
        'Median_StdDev_gray',
        'StdDev_Areas',
        'StdDev_Extents',
        'StdDev_Perimeters',
        'StdDev_ConvexAreas',
        'StdDev_Soliditys',
        'StdDev_Eccentricitys',
        'StdDev_MajorAxisLengths',
        'StdDev_MinorAxisLengths',
        'StdDev_EquivDiameters',
        'StdDev_Min_r',
        'StdDev_Max_r',
        'StdDev_Avr_r',
        'StdDev_Median_r',
        'StdDev_StdDev_r',
        'StdDev_Min_g',
        'StdDev_Max_g',
        'StdDev_Avr_g',
        'StdDev_Median_g',
        'StdDev_StdDev_g',
        'StdDev_Min_b',
        'StdDev_Max_b',
        'StdDev_Avr_b',
        'StdDev_Median_b',
        'StdDev_StdDev_b',
        'StdDev_Min_gray',
        'StdDev_Max_gray',
        'StdDev_Avr_gray',
        'StdDev_Median_gray',
        'StdDev_StdDev_gray',
        'Mode_Areas',
        'Mode_Extents',
        'Mode_Perimeters',
        'Mode_ConvexAreas',
        'Mode_Eccentricitys',
        'Mode_MajorAxisLengths',
        'Mode_MinorAxisLengths',
        'Mode_EquivDiameters',
        'Mode_Min_r',
        'Mode_Max_r',
        'Mode_Avr_r',
        'Mode_Median_r',
        'Mode_StdDev_r',
        'Mode_Min_g',
        'Mode_Max_g',
        'Mode_Avr_g',
        'Mode_Median_g',
        'Mode_StdDev_g',
        'Mode_Min_b',
        'Mode_Max_b',
        'Mode_Avr_b',
        'Mode_Median_b',
        'Mode_StdDev_b',
        'Mode_Min_gray',
        'Mode_Max_gray',
        'Mode_Avr_gray',
        'Mode_Median_gray',
        'Mode_StdDev_gray'
                ]