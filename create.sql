CREATE TABLE amostra (
	id INTEGER NOT NULL, 
	imagem VARCHAR NOT NULL, 
	classe VARCHAR(8) NOT NULL, 
	PRIMARY KEY (id)
)
CREATE TABLE atributo (
	id INTEGER NOT NULL, 
	amostra_id INTEGER, 
	nome VARCHAR NOT NULL, 
	valor FLOAT NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(amostra_id) REFERENCES amostra (id)
)
