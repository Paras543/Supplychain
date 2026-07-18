from pydantic import BaseModel


class ShipmentInput(BaseModel):
    Type: str
    Days_for_shipment_scheduled: int
    Benefit_per_order: float
    Sales_per_customer: float
    Category_Id: int
    Category_Name: str
    Customer_City: str
    Customer_Country: str
    Customer_Segment: str
    Customer_State: str
    Department_Id: int
    Department_Name: str
    Market: str
    Order_City: str
    Order_Country: str
    Order_Item_Discount: float
    Order_Item_Discount_Rate: float
    Order_Item_Product_Price: float
    Order_Item_Profit_Ratio: float
    Order_Item_Quantity: int
    Sales: float
    Order_Item_Total: float
    Order_Profit_Per_Order: float
    Order_Region: str
    Order_State: str
    Order_Status: str
    Product_Category_Id: int
    Product_Name: str
    Product_Price: float
    Product_Status: int
    Shipping_Mode: str
    order_date: str  # format: 'YYYY-MM-DD'


class RiskResponse(BaseModel):
    risk_score: float
    delay_probability: float
    prediction: str