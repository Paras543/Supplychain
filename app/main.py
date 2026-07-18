from fastapi import FastAPI, HTTPException
from app.schemas import ShipmentInput, RiskResponse
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.exception import CustomException
import sys

app = FastAPI(title="SupplyChainIQ", description="Shipment Delay Risk Scoring API")

pipeline = PredictionPipeline()  


@app.get("/")
def health_check():
    return {"status": "ok", "service": "SupplyChainIQ"}


@app.post("/predict", response_model=RiskResponse)
def predict_risk(shipment: ShipmentInput):
    try:
        raw_row = {
            'Type': shipment.Type,
            'Days for shipment (scheduled)': shipment.Days_for_shipment_scheduled,
            'Benefit per order': shipment.Benefit_per_order,
            'Sales per customer': shipment.Sales_per_customer,
            'Category Id': shipment.Category_Id,
            'Category Name': shipment.Category_Name,
            'Customer City': shipment.Customer_City,
            'Customer Country': shipment.Customer_Country,
            'Customer Segment': shipment.Customer_Segment,
            'Customer State': shipment.Customer_State,
            'Department Id': shipment.Department_Id,
            'Department Name': shipment.Department_Name,
            'Market': shipment.Market,
            'Order City': shipment.Order_City,
            'Order Country': shipment.Order_Country,
            'Order Item Discount': shipment.Order_Item_Discount,
            'Order Item Discount Rate': shipment.Order_Item_Discount_Rate,
            'Order Item Product Price': shipment.Order_Item_Product_Price,
            'Order Item Profit Ratio': shipment.Order_Item_Profit_Ratio,
            'Order Item Quantity': shipment.Order_Item_Quantity,
            'Sales': shipment.Sales,
            'Order Item Total': shipment.Order_Item_Total,
            'Order Profit Per Order': shipment.Order_Profit_Per_Order,
            'Order Region': shipment.Order_Region,
            'Order State': shipment.Order_State,
            'Order Status': shipment.Order_Status,
            'Product Category Id': shipment.Product_Category_Id,
            'Product Name': shipment.Product_Name,
            'Product Price': shipment.Product_Price,
            'Product Status': shipment.Product_Status,
            'Shipping Mode': shipment.Shipping_Mode,
            'order date (DateOrders)': shipment.order_date,
        }
        result = pipeline.predict(raw_row)
        return result

    except CustomException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


