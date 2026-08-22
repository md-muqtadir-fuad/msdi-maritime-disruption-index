# Semantic Event Pair Annotation Rubric & Guide
================================================================

This document defines the standard protocol for evaluating whether a pair of maritime news articles (**Article A** and **Article B**) report on the **same discrete physical disruption event** or **distinct events**.

---

## 1. Overview & Objective

In news-driven maritime disruption indices, multiple media outlets often cover the same underlying real-world shock (e.g., the *Ever Given* Suez blockage, an earthquake halting operations at Iskenderun Port, or a specific labor strike). 

The goal of this annotation task is to determine whether **Article A** and **Article B** belong to the same real-world event cluster.

---

## 2. The Core Decision Metric (`same_event`)

Every pair must be labeled as either **`1`** or **`0`**:

| Value | Label | Definition & Decision Criteria |
| :---: | :--- | :--- |
| **`1`** | **Same Event** | Both articles report on the **exact same underlying physical event, incident, or policy action** (same vessel, same port disruption, same strike, or same specific emergency shock), even if written by different journalists, published on different days within the event window, or providing ongoing updates on the same ongoing situation. |
| **`0`** | **Different Events** | The articles report on **different physical occurrences, different locations, distinct incidents in different time periods**, or merely share general industry themes (e.g., both mention container shipping or general inflation) without describing the same specific real-world event. |

---

## 3. Detailed Decision Rules & Real-World Examples

### **Rule 1: Exact Wire Reprints / Syndications $\implies 1$ (Same Event)**
* **Example**: Reuters publishes an article about *"Singapore Port allowing night barge movements to reduce Pasir Panjang backlogs"*, and *gCaptain* / *The Maritime Executive* repost the story.
* **Verdict**: **`same_event = 1`**

### **Rule 2: Multi-Outlet Coverage of the Same Real-World Shock $\implies 1$ (Same Event)**
* **Example**: 
  * *Article A*: BBC reports on the *US & UK air strikes targeting Houthi sites in Yemen on Jan 12, 2024*.
  * *Article B*: The Guardian reports *Oil prices hit $80/barrel following US-UK strikes on Red Sea Houthi targets on Jan 12, 2024*.
* **Verdict**: **`same_event = 1`** (Same underlying military-maritime event causing Red Sea diversions).

### **Rule 3: Developing Updates on the Same Specific Incident $\implies 1$ (Same Event)**
* **Example**: 
  * *Article A (March 25)*: *Ever Given container ship stuck in Egypt's Suez Canal blocking traffic*.
  * *Article B (March 29)*: *Tugboats partially refloat Ever Given in Suez Canal after nearly a week*.
* **Verdict**: **`same_event = 1`** (Both describe the same discrete blockage event).

### **Rule 4: Same Incident at a Specific Port/Terminal $\implies 1$ (Same Event)**
* **Example**:
  * *Article A*: DW reports *Large fire put out at earthquake-damaged Iskenderun Port in Turkey*.
  * *Article B*: CNA reports *Fire shuts down operations at Turkey's Iskenderun Port following 7.8-magnitude earthquake*.
* **Verdict**: **`same_event = 1`**

---

### **Rule 5: Distinct Incidents at Different Ports/Locations $\implies 0$ (Different Events)**
* **Example**:
  * *Article A*: Labor strike halts container handling at the *Port of Liverpool* in September.
  * *Article B*: Longshore workers strike at the *Port of Felixstowe* in October.
* **Verdict**: **`same_event = 0`** (Different ports, different worker unions, different physical events).

### **Rule 6: Recurring Military or Weather Incidents on Different Dates $\implies 0$ (Different Events)**
* **Example**:
  * *Article A (Jan 12)*: US-UK launch first round of air strikes on Yemen in response to Red Sea attacks.
  * *Article B (Feb 24)*: US-UK launch fourth fresh round of joint strikes on 18 Houthi sites.
* **Verdict**: **`same_event = 0`** (Distinct military strikes occurring over a month apart).

### **Rule 7: General Topic Overlap Without Specific Shared Event $\implies 0$ (Different Events)**
* **Example**:
  * *Article A*: General commentary on *how global inflation affects UK households and fuel companies*.
  * *Article B*: *Maersk orders green methanol container vessels for 2040 carbon neutrality*.
* **Verdict**: **`same_event = 0`** (Both mention shipping/energy, but have zero shared event).

---

## 4. Confidence & Notes Columns

* **`confidence`**: Record your level of certainty in the decision:
  * `High`: Clear, unambiguous match or distinct non-match.
  * `Medium`: Moderate uncertainty (e.g., unclear whether an article is a direct continuation or a separate minor follow-up).
  * `Low`: Insufficient details in the text excerpts to be confident.
* **`notes`**: (Optional) Brief rationale for borderline or difficult edge cases.
