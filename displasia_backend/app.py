from fastapi import FastAPI


from fastapi.staticfiles import StaticFiles
from .models import Classe, Amostra, Atributo
from .services import model
from .db.repository import amostra as amostra_repo

app = FastAPI()

app.mount("/static", StaticFiles(directory="displasia_backend/static"), name="static")

#app.dr = dataReader.dataReader(FeaturesTypesUsed = 'all')
#app.data, app.target = app.dr.GetData()
#app.data.columns = app.dr.engNames
#app.train_data, app.test_data, app.train_target, app.test_target = train_test_split(app.data, app.target, train_size = 0.8, random_state = 1)
@app.get("/")
async def root():
    model.train()
    
    return {"message": "Hello World"}

@app.get("/pred/{id}")
async def predict(id: int):
    model.predict(id)
    return {"message": "Hello World"}

@app.get("/amostras")
async def amostras():
    return amostra_repo.get_all_amostras()

@app.get("/amostras/{id}")
async def amostras_by_id(id: int):
    return amostra_repo.get_amostra_by_id(id)

@app.get("/amostras/{id}/local")
async def amostras_local_exp(id: int):
    amostra = amostra_repo.get_amostra_by_id(id)
    if(amostra):
        print(1)#local_exp = 

@app.post("/amostras/{id}/reclassificar")
async def reclassificar():
    print(1)