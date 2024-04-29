# Source: https://firebase.google.com/docs/firestore/quickstart
import datetime
from typing import Literal
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

from LicensePlateDetection.utils.formatDate import formatDate

# Load service account credentials
cred = credentials.Certificate('serviceAccount.json')

app = firebase_admin.initialize_app(cred)

db = firestore.client()

# ==============================================================================
# Entities
# ==============================================================================

class DetectionsEntity:
    document_name = "detections"

    def __init__(self, id: str, license_plate: str, date_detected: datetime.datetime, registered: str):
        self.id: str = id
        self.license_plate: str = license_plate
        self.date_detected: str = date_detected
        self.formatted_date_detected: str = formatDate(date_detected)
        self.registered: Literal['Registered', 'Unregistered'] = registered

    def from_dict(data_dict: dict):
        """Static Method: JSON -> Entity"""
        return DetectionsEntity(
            id=data_dict["id"],
            license_plate=data_dict["license_plate"],
            date_detected=data_dict["date_detected"],
            registered=data_dict["registered"]
        )

    def to_dict(self):
        """Method: Entity -> JSON"""
        return {
            "id": self.id,
            "license_plate": self.license_plate,
            "date_detected": self.date_detected,
            "formatted_date_detected": self.formatted_date_detected,
            "registered": self.registered
        }

class RegisteredLicensePlateEntity:
    document_name = "registered_license_plates"

    def __init__(self, id: str, license_plate: str, date_registered: datetime.datetime):
        self.id = id
        self.license_plate = license_plate
        self.date_registered = date_registered
    
    def from_dict(data_dict: dict):
        return RegisteredLicensePlateEntity(
            id=data_dict["id"],
            license_plate=data_dict["license_plate"],
            date_registered=data_dict["date_registered"]
        )

    def to_dict(self):
        return {
            "id": self.id,
            "license_plate": self.license_plate,
            "date_registered": self.date_registered
        }

# ==============================================================================
# Repository Pattern for the database (READS)
# ==============================================================================

# 1. Get all detections
def getAllDetections() -> list[DetectionsEntity]: 
    license_plates_refs = db.collection(DetectionsEntity.document_name)
    docs = license_plates_refs.stream()

    # 1. DO -> Entity 
    license_plate_entities: list[DetectionsEntity] = []
    
    # 2. Convert each document to a LicensePlateModel instance
    for doc in docs:
        doc_dict = doc.to_dict()
        license_plate_entities.append(DetectionsEntity.from_dict(doc_dict))

    return license_plate_entities

# 2. Get all registered license plates
def getAllRegisteredLicensePlates() -> list[RegisteredLicensePlateEntity]:
    license_plates_refs = db.collection(RegisteredLicensePlateEntity.document_name)
    docs = license_plates_refs.stream()

    # 1. DO -> Entity 
    license_plate_entities: list[RegisteredLicensePlateEntity] = []
    
    # 2. Convert each document to a LicensePlateModel instance
    for doc in docs:
        doc_dict = doc.to_dict()
        license_plate_entities.append(RegisteredLicensePlateEntity.from_dict(doc_dict))

    return license_plate_entities

# 3. Get specific registered license plate
def getRegisteredLicensePlate(license_plate: str) -> RegisteredLicensePlateEntity:
    doc_ref = db.collection(RegisteredLicensePlateEntity.document_name).document(license_plate)
    doc = doc_ref.get()

    if not doc.exists:
        return None

    doc_dict = doc.to_dict()
    return RegisteredLicensePlateEntity.from_dict(doc_dict)

# ==============================================================================
# Repository Pattern for the writes (WRITES)
# ==============================================================================

# 2. Add new license plate detection
def insertNewDetection(license_plate: str, date_detected: datetime.datetime):
    # 1. Check if license plate is registered
    existingLicensePlate = getRegisteredLicensePlate(license_plate)
    
    # 2. Determine the Registered literal
    registered = "Registered" if existingLicensePlate != None else "Unregistered"

    # 4. Create the new detection entity to be added. (No id, because it will be randomly generated).
    doc_ref = db.collection(DetectionsEntity.document_name).document() # Not passing in document will generate a random id.
    
    new_detection = DetectionsEntity(doc_ref.id, license_plate, date_detected, registered)

    # 5. DB operation to insert.
    doc_ref.set(new_detection.to_dict())

    return new_detection

# 3. Insert new registered license plate
def insertNewRegisteredLicensePlate(license_plate: str, date_registered: datetime.datetime):
    # 1. Create the new registered license plate entity to be added.
    new_registered_license_plate = RegisteredLicensePlateEntity(id=license_plate, license_plate=license_plate, date_registered=date_registered)
    
    # 2. DB operation to insert.
    doc_ref = db.collection(RegisteredLicensePlateEntity.document_name).document(license_plate) # The id will also be the license plate as it's unique.
    doc_ref.set(new_registered_license_plate.to_dict())
