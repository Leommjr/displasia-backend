from sqlmodel import Session, select
from .db.database import create_db_and_tables, engine
from .models import Classe, Amostra, Atributo
import csv


def get_classe(classe_str):
    return Classe(classe_str)

def create_data():
    path = 'displasia_backend/Data/Displasia/'
    csv_files = ['caracteristicas-healthy.csv', 'caracteristicas-mild.csv', 'caracteristicas-moderate.csv', 'caracteristicas-severe.csv']
    for csv_file in csv_files:
        with open(path+csv_file, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            amostras = []

            for row in reader:
                # Process each row from the CSV
                amostra_data = {
                    'imagem': row['Imagem'],
                    'classe': Classe(row['Classe']),
                }
                amostra = Amostra(**amostra_data)

                # Process attribute columns
                for key, value in row.items():
                    if key not in ('Imagem', 'Classe'):
                        atributo_data = {
                            'nome': key,
                            'valor': float(value),
                        }
                        atributo = Atributo(**atributo_data)
                        amostra.atributos.append(atributo)

                amostras.append(amostra)

            # Add the amostras to the database
            with Session(engine) as session:
                session.add_all(amostras)
                session.commit()
    # Verify that the data has been inserted
    with Session(engine) as session:
        amostras = session.exec(select(Amostra)).all()
        for amostra in amostras:
            print(f"Amostra: {amostra.imagem}, Classe: {amostra.classe}")
            for atributo in amostra.atributos:
                print(f"  Atributo: {atributo.nome}, Valor: {atributo.valor}")

def select_amostras():
    with Session(engine) as session:
        statement = select(Amostra)
        results = session.exec(statement)
        for amostra in results:
            print(amostra)





def main():
    create_db_and_tables()
    create_data()
    #create_amostras()


if __name__ == "__main__":
    main()