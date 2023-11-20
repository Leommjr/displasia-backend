from sqlmodel import Session, select
from typing import List
from ..database import create_db_and_tables, engine
import pandas as pd
from ...models import Classe, Amostra, Atributo

def get_all_amostras() -> List[Amostra]:
    with Session(engine) as session:
        statement = select(Amostra)
        results = session.exec(statement)
        amostras = results.all()
    return amostras


def get_amostra_by_id(id: int) -> Amostra:
    with Session(engine) as session:
        statement = select(Amostra).where(Amostra.id == id)
        results = session.exec(statement)
        amostra = results.first()
    return amostra

def get_amostra_by_id_dt(id: int) -> Amostra:
    with Session(engine) as session:
        final_data = pd.Series(dtype=float)
        statement = select(Amostra).where(Amostra.id == id)
        results = session.exec(statement)
        amostra = results.first()
        stmt = select(Atributo.amostra_id, Atributo.nome, Atributo.valor).where(amostra.id == Atributo.amostra_id)
        res = session.exec(stmt)
        for at in res:
            final_data = pd.concat([final_data, pd.Series({at.nome: at.valor}, dtype=float)])
    return final_data

def get_amostras_by_classe_dt(classe: str) -> List[Amostra]:
    with Session(engine) as session:
        columns = ['id', 'imagem', 'classe']

        # Create a query to select the pivot columns and join 'amostra' with 'atributo'
        
        statement = select(Amostra.id, Amostra.imagem, Amostra.classe).where(Amostra.classe == classe)
        results = session.exec(statement)
        amostra = results.first()
        stmt = select(Atributo.amostra_id, Atributo.nome, Atributo.valor).where(amostra.id == Atributo.amostra_id)
        res = session.exec(stmt)
        for at in res:
            columns.append(at.nome)
        final_data = pd.DataFrame(columns=columns)
        results = session.exec(statement)
        for amt in results:
            row = []
            row.append(amt.id)
            row.append(amt.imagem)
            row.append(amt.classe.value)
            stmt = select(Atributo.amostra_id, Atributo.nome, Atributo.valor).where(amt.id == Atributo.amostra_id)
            res = session.exec(stmt)
            for at in res:
                row.append(at.valor)
            final_data = pd.concat([final_data, pd.DataFrame([row], columns=columns)], ignore_index=True)
    return final_data
