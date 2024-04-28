# Source: https://firebase.google.com/docs/firestore/quickstart
import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Load service account credentials
cred = credentials.Certificate('serviceAccount.json')

app = firebase_admin.initialize_app(cred)

db = firestore.client()

# ==============================================================================
# Entities
# ==============================================================================

class LicensePlateEntity:
    def __init__(self, id, license_plate, date_detected):
        self.id = id
        self.license_plate = license_plate
        self.date_detected = date_detected

    def to_dict(self):
        return {
            "id": self.id,
            "license_plate": self.license_plate,
            "date_detected": self.date_detected
        }


# ==============================================================================
# Repository Pattern for the database (READS)
# ==============================================================================

# 1. Get all
def getAllLicensePlates() -> list[LicensePlateEntity]: 
    license_plates_refs = db.collection('license_plates')
    docs = license_plates_refs.stream()

    # TODO: Step 1 and 2 might actually not be necessary because the model is very basic (no entity logic) so the docs itself matches the entity atm.

    # 1. DO -> Entity 
    license_plate_entities = []
    
    # 2. Convert each document to a LicensePlateModel instance
    for doc in docs:
        license_plate_entities.append(LicensePlateEntity(doc.id, doc.get("license_plate"), doc.get("date_detected")))

    return license_plate_entities

# ==============================================================================
# Repository Pattern for the writes (WRITES)
# ==============================================================================

# 2. Add new data
def insertNewLicensePlate(license_plate: str, date_detected: datetime.datetime):
    doc_ref = db.collection("license_plates").document() # Not passing in document will generate a random id.
    doc_ref.set({"license_plate": license_plate, "date_detected": date_detected})