
##  Summer Tehqiq Research Program 1 (STRP1) – Habib University  
**Supervisor:** *Prof. SM Hur Rizvi*  

This project was undertaken as part of the **Summer Tehqiq Research Program 1** (STRP1) at **Habib University**, where I worked as an undergraduate researcher under the supervision of **Prof. SM Hur Rizvi**.

---

##  **Motivation**

With the integration of **Photovoltaic (PV) systems** and **Battery Energy Storage Systems (BESS)**, modern distribution networks are experiencing a shift from unidirectional to **bidirectional electricity flow**. Traditional grid structures were designed for one-way power flow—from centralized plants to consumers—which poses significant challenges for **parameter estimation** and **grid management**.

---

## **Objective**

The goal of this project was to develop a **data-driven model** capable of learning and predicting grid behavior under various operating conditions. Using the **IEEE 33-bus system**, we simulated two main scenarios:

1. **Loads Only**  
2. **Loads + Distributed Energy Resources (DERs)** including PV and BESS  

These simulations provided a rich dataset for training machine learning models and building **reduced equivalent networks** from estimated parameters.

---

##  **Repository Structure**

```bash
.
├── Branch6-18 3.dss             # OpenDSS simulation file with PV and BESS
├── Onlyloadsfinal1.dss         # OpenDSS simulation file with loads only
├── PVBESS.py                   # Automates dataset extraction and exports data to Excel
├── PVBESS_pred.py              # ML model for predicting PV and BESS parameters
├── Reduced Model (1).dss       # Aggregated equivalent network from loads-only estimation
├── loadsonly.py                # Dataset extraction for the loads-only scenario
├── loadsonly_pred.py           # ML model for loads-only parameter estimation
├── Inmic_conference.pdf        # Research paper presented at INMIC 2025
├── strp_final_report.pdf       # Final comprehensive project report
└── README.md                   # Project overview and documentation
```
