import argparse

parser = argparse.ArgumentParser(description="Qui êtes-vous?")
parser.add_argument(
    "--prenom", type=str, default="Toto", help="Un prénom à afficher"
)
args = parser.parse_args()
print(args.prenom)
