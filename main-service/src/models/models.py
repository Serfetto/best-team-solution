from datetime import datetime
import uuid
from sqlalchemy import JSON, UUID, Boolean, Column, Integer, String, ForeignKey, TIMESTAMP
from sqlalchemy.orm import relationship
from src.models.database import Base

class User(Base):
    __tablename__ = 'users'

    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    email = Column(String, nullable=False, unique=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, server_default="0")
    is_admin = Column(Boolean, server_default="0")
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

    def __str__(self):
        return self.email
    
class ProductServices(Base):
    __tablename__ = 'product_services_secondary'

    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    product_id = Column(UUID, ForeignKey('products.id'))
    services_id = Column(UUID, ForeignKey('services.id'))
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

    def __str__(self):
        return self.id

class Services(Base):
    __tablename__ = 'services'

    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    products = relationship('Product', secondary="product_services_secondary", back_populates='services')
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    def __str__(self):
        return self.name
    
class Product(Base):
    __tablename__ = 'products'

    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    services = relationship('Services', secondary="product_services_secondary", back_populates='products')
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

    def __str__(self):
        return self.name
    
class Reviews(Base):
    __tablename__ = 'reviews'

    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    text = Column(String, nullable=False)
    sentiment_label = Column(String, nullable=True)
    timestamp = Column(TIMESTAMP, nullable=True)
    product = Column(String, nullable=True)
    source = Column(String, nullable=True)
    rating = Column(Integer, nullable=True)
    review_theses = Column(JSON, nullable=True)
    convenience_and_technology_ratings = Column(JSON, nullable=True)
    emotional_words = Column(JSON, nullable=True)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

    def __str__(self):
        return self.id

class OldReviews(Base):
    __tablename__ = 'oldreviews'

    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    agentAnswerText = Column(String, nullable=True)
    dateCreate = Column(TIMESTAMP, nullable=True)
    grade = Column(Integer, nullable=True)
    text = Column(String, nullable=True)
    title = Column(String, nullable=True)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

    def __str__(self):
        return self.id