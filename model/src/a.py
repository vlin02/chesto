import json
import torch
from pymongo import MongoClient
from input import load_lookup, batch_states, decode_state
from net import NN

S = """{"done":false,"p1":{"reward":0,"state":{"ally":{"team":{"Arcanine":[1,0.48,0.48,0.3883333333333333,0.305,0.36,0.305,0.3466666666666667,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],"Slowking":[0,0.5166666666666667,0.5166666666666667,0.22833333333333333,0.31833333333333336,0.37666666666666665,0.4066666666666667,0.17166666666666666,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0],"Bastiodon":[0,0.42,0.42,0.16166666666666665,0.5833333333333334,0.22333333333333333,0.49333333333333335,0.17333333333333334,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0],"Swalot":[0,0.5433333333333333,0.5433333333333333,0.305,0.335,0.305,0.335,0.25,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"Cinccino":[0,0.43333333333333335,0.43333333333333335,0.3416666666666667,0.245,0.26,0.245,0.3983333333333333,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"Veluza":[0,0.4866666666666667,0.4866666666666667,0.37,0.28833333333333333,0.3016666666666667,0.265,0.28,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0]},"active":"Arcanine","x":[6,0,0,0,0,6,0]},"foe":{"team":{"Registeel":[1,0.43666666666666665,0.43666666666666665,0.28,0.48333333333333334,0.28,0.48333333333333334,0.21333333333333335,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]},"active":"Registeel","x":[6,0,0,0,0,6,0]},"option":{"tera":true,"moves":["Wild Charge","Close Combat","Flare Blitz","Morning Sun"],"switches":["Slowking","Bastiodon","Swalot","Cinccino","Veluza"]}}}}"""

if __name__ == "__main__":
    device = torch.device("cpu")

    DB_URL = "mongodb://admin:4wj62MDCv%25X%5ErU3F@172.31.30.235:27017/"
    client = MongoClient(DB_URL)
    lookup = load_lookup(client["chesto"], device)
    nn = NN(lookup).to(device)
    update = json.loads(S)

    print(
        nn(
            batch_states(
                [
                    decode_state(
                        lookup, update["p1"]["state"], device=torch.device("cpu")
                    )
                ]
            )
        )
    )
