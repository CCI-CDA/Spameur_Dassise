from faker import Faker
import csv
import random

fake = Faker('fr_FR')

# Define examples of French spam messages
spam_messages = [
    "Félicitations! Vous avez gagné un billet gratuit.",
    "Offre exclusive! Achetez maintenant et obtenez 50% de réduction.",
    "Votre compte a été suspendu. Cliquez ici pour réactiver.",
    "Vous avez un nouveau message en attente. Cliquez ici pour lire.",
    "Obtenez un iPhone gratuit en cliquant sur ce lien.",
    "Gagnez une carte cadeau de 1000€ maintenant!",
    "Offre limitée, agissez maintenant!",
    "Vous avez été sélectionné pour un prix spécial.",
    "Gagnez de l'argent facilement depuis chez vous.",
    "Urgent: Votre compte nécessite une vérification."
]

# Generate 1 million entries (500,000 spam and 500,000 non-spam)
num_entries = 5000000
data = []

for _ in range(num_entries // 2):
    data.append([random.choice(spam_messages), 1])
    data.append([fake.sentence(), 0])

# Write the data to a CSV file
with open('spam_data.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['message', 'label'])
    writer.writerows(data)