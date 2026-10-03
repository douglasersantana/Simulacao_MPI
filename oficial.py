from mpi4py import MPI
import json
import random
import time
from collections import deque
from dataclasses import dataclass

# Executar:
# & "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe" -n 3 python oficial.py

# Inicializa o comunicador MPI.
# COMM_WORLD reúne todos os processos participantes da execução.
comm = MPI.COMM_WORLD

# Identificador único do processo (rank).
# Rank 0 = Origem
# Rank 1 = Servidor
# Rank 2 = Destino
rank = comm.Get_rank()

# Quantidade total de processos MPI.
size = comm.Get_size()

# O trabalho necessita obrigatoriamente de três processos.
if size < 3:
    if rank == 0:
        print(f"Este programa precisa ser executado com pelo menos 3 processos MPI. Foram encontrados {size}.")
    comm.Abort(1)

DESTINO = 2
SERVIDOR = 1
ORIGEM = 0

NUM_MENSAGENS = 5

# Arquivo utilizado como armazenamento persistente das mensagens.
# Simula o funcionamento de um broker de mensagens.
ARQUIVO_SERVIDOR = "mensagens_servidor.txt"


@dataclass
class Mensagem:
    seq: int
    conteudo: str
    origem: int


def dormir():
    # Simula atrasos imprevisíveis da rede/processamento.
    # Mesmo dormindo, as mensagens permanecem persistidas.
    tempo = random.uniform(0.5, 2.5)
    print(f"[{rank}] Dormindo {tempo:.2f}s")
    time.sleep(tempo)


def salvar_mensagem(msg: Mensagem):
    # Persistência da mensagem em disco.
    # Caso o processo fique indisponível, a mensagem continua armazenada.
    with open(ARQUIVO_SERVIDOR, "a", encoding="utf-8") as arquivo:
        arquivo.write(json.dumps({
            "seq": msg.seq,
            "conteudo": msg.conteudo,
            "origem": msg.origem
        }, ensure_ascii=False) + "\n")


def recuperar_mensagem(seq: int) -> Mensagem | None:
    # Recupera a mensagem persistida antes de encaminhá-la ao destino.
    with open(ARQUIVO_SERVIDOR, "r", encoding="utf-8") as arquivo:
        for linha in arquivo:
            dados = json.loads(linha)
            if dados["seq"] == seq:
                return Mensagem(**dados)
    return None



# PROCESSO DE ORIGEM (Rank 0)

if rank == ORIGEM:

    # Fila persistente local da origem.
    fila = deque()

    # Lista das operações MPI assíncronas.
    requests = []

    print("\nORIGEM INICIADA\n")

    # Produz todas as mensagens antes do envio.
    for i in range(1, NUM_MENSAGENS + 1):
        fila.append(
            Mensagem(
                seq=i,
                conteudo=f"Mensagem {i}",
                origem=rank
            )
        )

    while fila:

        dormir()

        msg = fila.popleft()

        # Comunicação NÃO BLOQUEANTE.
        # O processo continua executando enquanto o MPI realiza o envio.
        req = comm.isend(msg, dest=SERVIDOR, tag=msg.seq)

        requests.append(req)

        print(f"[Origem] Enviou {msg.conteudo}")

    # Mensagem especial indicando fim da transmissão.
    req = comm.isend(None, dest=SERVIDOR, tag=0)
    requests.append(req)

    # Aguarda a conclusão de todos os envios.
    for req in requests:
        req.wait()



# PROCESSO SERVIDOR (Rank 1)

elif rank == SERVIDOR:

    # Fila contendo apenas os identificadores das mensagens.
    # O conteúdo será recuperado do arquivo persistente.
    fila = deque()

    print("\nSERVIDOR INICIADO\n")

    # Limpa o arquivo antes de iniciar uma nova execução.
    open(ARQUIVO_SERVIDOR, "w", encoding="utf-8").close()

    while True:

        # Recepção NÃO BLOQUEANTE.
        recv_req = comm.irecv(source=ORIGEM, tag=MPI.ANY_TAG)

        msg = recv_req.wait()

        if msg is None:
            break

        # A mensagem é persistida antes do encaminhamento.
        salvar_mensagem(msg)

        fila.append(msg.seq)

        print(f"[Servidor] Recebeu {msg.conteudo}")

    while fila:

        dormir()

        seq = fila.popleft()

        # Recupera a mensagem persistida.
        msg = recuperar_mensagem(seq)

        if msg is None:
            continue

        # Encaminhamento assíncrono ao destino.
        req = comm.isend(msg, dest=DESTINO, tag=seq)

        req.wait()

        print(f"[Servidor] Encaminhou {msg.conteudo}")

    # Sinaliza o encerramento ao destino.
    req = comm.isend(None, dest=DESTINO, tag=0)

    req.wait()



# PROCESSO DESTINO (Rank 2)

elif rank == DESTINO:

    # Fila local do destino.
    fila = deque()

    print("\nDESTINO INICIADO\n")

    total = 0

    while True:

        # Recepção assíncrona das mensagens.
        recv_req = comm.irecv(source=SERVIDOR, tag=MPI.ANY_TAG)

        msg = recv_req.wait()

        if msg is None:
            break

        fila.append(msg)

        dormir()

        entregue = fila.popleft()

        total += 1

        print(f"[Destino] Recebeu {entregue.conteudo}")

    print(f"\nTotal recebido: {total}")