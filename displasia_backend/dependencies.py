"""
FastApi dependencies module
"""

from fastapi import Request
#from .db.database import SessionLocal
from .logger import log


"""async def get_db() -> SessionLocal:
    
    get sessao banco de dados
    @return: sessao no banco de dados
    
    data_base = SessionLocal()
    try:
        yield data_base
    finally:
        data_base.close()
"""
async def log_access(req: Request) -> None:
    """
    Dependencia para logar todos os acessos da api
    @param req: Objeto representando a requisicao HTTP
    @return: None
    """
    log.warning("Acesso %s    %s    %s", req.method, req.url.path, req.client.host)