# Maritime Disruption Scoring Rubric & Annotation Guide
================================================================

This document defines the standard annotation protocol for evaluating maritime news articles on an **ordinal scale from 0 to 4**.

---

## 1. Relevance Label (`relevance_label`)
* `relevant`: The article explicitly discusses container shipping, maritime logistics, ports, vessel movements, freight rates, chokepoints, or supply chain disruptions.
* `irrelevant`: General economics, stock commentary, road trucking only, passenger cruise news, or non-supply-chain stories.
* `uncertain`: Ambiguous context where maritime relevance is tangential.

---

## 2. The Standard 0–4 Disruption Severity Scale

Every dimension must be scored as an integer from **0 to 4**:

| Score | Severity Level | Operational Definition & Criteria |
| :---: | :--- | :--- |
| **0** | **No Disruption / Irrelevant** | Normal market conditions, routine business operations, new service launch, company earnings, or irrelevant text. |
| **1** | **Low / Localized Friction** | Minor delays (1-2 days), localized weather advisories, minor administrative fee changes, slight schedule adjustments with no ripple effect. |
| **2** | **Moderate Operational Disruption** | Measurable operational bottlenecks: localized berth congestion, localized labor disputes, selective port slow-downs, regional blank sailings. |
| **3** | **High Regional / Multi-Actor Disruption** | Widespread severe port congestion across major hubs, significant labor strikes, widespread container equipment shortages, sharp spot rate volatility. |
| **4** | **Severe Systemic / Critical Chokepoint Shock** | Catastrophic disruptions: Suez Canal obstruction, Red Sea missile rerouting around Cape, major multi-month COVID port shutdowns, structural fleet capacity collapse. |

---

## 3. The 7 Disruption Taxonomy Criteria (`K = 7`)

1. **`port_disruption_score`**: Berth congestion, vessel queues at terminals, longshoremen strikes, crane breakdowns, container yard dwell time spikes.
2. **`global_container_movement_disruption_score`**: Chokepoint blockages (Suez/Panama/Bab-el-Mandeb), vessel diversions around Cape, carrier blank sailings, transit delays.
3. **`freight_cost_disruption_score`**: Container spot rate surges (SCFI spikes), war-risk surcharges, emergency bunker surcharges, container lease fee spikes.
4. **`supply_disruption_score`**: Upstream component shortages, raw material delays, inventory stockouts, supply delivery bottlenecks.
5. **`production_disruption_score`**: Factory shutdowns, manufacturing halts, assembly line stoppages caused by missing shipping containers.
6. **`covid19_related_disruption_score`**: Quarantine closures, crew change restrictions, zero-COVID port halts, pandemic vessel delays.
7. **`demand_disruption_score`**: Sudden consumer demand surge/collapse, holiday front-loading, extreme purchasing volatility.

---

## 4. Overall & Confidence Columns

* **`overall_disruption_score` (0-4)**: The overall perceived disruption shock across the entire article.
* **`dominant_dimension`**: The primary dimension among the 7 that is most affected (e.g., `port_disruption_score`).
* **`annotator_confidence`**: `high`, `medium`, or `low`.
