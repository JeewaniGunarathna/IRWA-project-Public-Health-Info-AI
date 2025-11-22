# backend/retrieval_agent/agent.py
from __future__ import annotations

from typing import Any, Dict, List
import re
import requests
import spacy
from urllib.parse import quote  # <-- needed
from .adapters.base import FactsAdapter
from .adapters.cdc_flu import CDCFluAdapter
from .adapters.worldbank import WorldBankAdapter
from .nlp import parse_query

nlp = spacy.load("en_core_web_sm")


# Small country-name -> ISO2 fallback mapping for disease.sh
_ISO2_MAP = {
    "sri lanka": "LK",
    "india": "IN",
    "united states": "US",
    "u.s.": "US",
    "us": "US",
    "usa": "US",
    "united kingdom": "GB",
    "uk": "GB",
    "bangladesh": "BD",
    "pakistan": "PK",
    "nepal": "NP",
    "china": "CN",
    "japan": "JP",
    "australia": "AU",
    "canada": "CA",
    "france": "FR",
    "germany": "DE",
    "spain": "ES",
    "italy": "IT",
    "indonesia": "ID",
}

def _to_iso2_or_original(name: str) -> str:
    if not name:
        return name
    s = name.strip().lower()
    return _ISO2_MAP.get(s, name)


# --- add near top of agent.py (module level, not inside the class) ---
_COUNTRY_ISO2 = {
    "sri lanka": "lk",
    "india": "in",
    "united states": "us",
    "u.s.": "us",
    "usa": "us",
    "us": "us",
    "united kingdom": "gb",
    "uk": "gb",
    "pakistan": "pk",
    "bangladesh": "bd",
    "nepal": "np",
    "china": "cn",
    "japan": "jp",
    "australia": "au",
    "canada": "ca",
    "france": "fr",
    "germany": "de",
    "spain": "es",
    "italy": "it",
    "indonesia": "id",
    "world": "world",
}

def _iso2_or_none(name: str) -> str | None:
    if not name:
        return None
    s = name.strip().lower()
    if len(s) in (2, 3) and s.isalpha():
        return s.lower()
    return _COUNTRY_ISO2.get(s)

def _owid_slug(name: str) -> str:
    # simple slug for OWID country pages
    return (name or "").strip().lower().replace(" ", "-")


class InformationRetrievalAgent:
    def __init__(self):
        self.covid_api = "https://disease.sh/v3/covid-19/countries/"
        self.medicine_api = "https://api.fda.gov/drug/event.json?search=patient.drug.medicinalproduct:"
        self.usda_api = "https://api.nal.usda.gov/fdc/v1/foods/search"
        # NOTE: consider reading from env instead of hard-coding
        self.usda_api_key = "yUHeDOCO81Ht367aOuHLyppQN8zlnYrGWh7nA4Gp"

        self.known_medicines = ["ibuprofen", "paracetamol", "aspirin", "acetaminophen", "amoxicillin"]
        self.known_diseases = ["covid", "dengue", "malaria", "diabetes", "hypertension", "flu", "influenza"]

        # Adapters (order matters)
        self.adapters: List[FactsAdapter] = [
            CDCFluAdapter(),
            WorldBankAdapter(),
        ]

    # ----------------- Keyword extraction -----------------
    def extract_keywords(self, question: str) -> Dict[str, Any]:
        doc = nlp(question)
        disease = None
        medicine = None
        info_type = None
        country = None

        for ent in doc.ents:
            if ent.label_ == "GPE":
                country = ent.text

        ql = question.lower()
        for d in self.known_diseases:
            if d in ql:
                disease = d
                break

        for m in self.known_medicines:
            if m in ql:
                medicine = m
                break

        if any(word in ql for word in ["side effect", "adverse", "reaction"]):
            info_type = "side_effects"
        elif any(word in ql for word in ["cases", "infected", "infection"]):
            info_type = "cases"
        elif any(word in ql for word in ["death", "deaths"]):
            info_type = "deaths"
        elif any(word in ql for word in ["recovered"]):
            info_type = "recovered"
        elif any(word in ql for word in ["treatment", "dose", "dosage"]):
            info_type = "treatment"
        elif any(word in ql for word in ["symptoms"]):
            info_type = "symptoms"
        elif any(word in ql for word in ["nutrition", "diet", "food", "vitamin"]):
            info_type = "nutrition"
        elif any(word in ql for word in ["habit", "exercise", "healthy"]):
            info_type = "healthy_habits"
        else:
            info_type = "general"

        return {
            "question": question,
            "disease": disease,
            "medicine": medicine,
            "info_type": info_type,
            "country": country,
            "region": country,  # normalize key others use
        }

    # ----------------- Built-in fetchers -----------------
    def fetch_covid_data(self, country="World", info_type="all"):
        """
        Returns a normalized payload:
          { "type": "covid_*", "summary": "...", "data": {...} }
        """
        country_enc = quote(str(country).strip())
        url = f"{self.covid_api}{country_enc}"

        try:
            r = requests.get(url, timeout=20)
            if r.status_code >= 400:
                iso2 = _to_iso2_or_original(country)
                url_iso = f"{self.covid_api}{quote(iso2)}?strict=true"
                r = requests.get(url_iso, timeout=20)

            r.raise_for_status()
            data = r.json()

            if info_type == "cases":
                return {
                    "type": "covid_cases",
                    "summary": f"{data.get('country', country)} — {data.get('cases')} total COVID cases ({data.get('todayCases', 0)} today).",
                    "data": {"country": data.get("country", country),
                             "cases": data.get("cases"),
                             "todayCases": data.get("todayCases", 0)},
                    "sources": [{"name": "disease.sh (JHU)", "url": "https://disease.sh/"}],
                }
            elif info_type == "deaths":
                return {
                    "type": "covid_deaths",
                    "summary": f"{data.get('country', country)} — {data.get('deaths')} total COVID deaths ({data.get('todayDeaths', 0)} today).",
                    "data": {"country": data.get("country", country),
                             "deaths": data.get("deaths"),
                             "todayDeaths": data.get("todayDeaths", 0)},
                    "sources": [{"name": "disease.sh (JHU)", "url": "https://disease.sh/"}],
                }
            elif info_type == "recovered":
                return {
                    "type": "covid_recovered",
                    "summary": f"{data.get('country', country)} — {data.get('recovered')} recovered.",
                    "data": {"country": data.get("country", country),
                             "recovered": data.get("recovered")},
                    "sources": [{"name": "disease.sh (JHU)", "url": "https://disease.sh/"}],
                }

            # default: include full data but still normalized
            return {
                "type": "covid_all",
                "summary": f"COVID-19 data for {data.get('country', country)} retrieved.",
                "data": data,
                "sources": [{"name": "disease.sh (JHU)", "url": "https://disease.sh/"}],
            }

        except Exception as e:
            return {
                "type": "covid_error",
                "summary": f"COVID-19 data for {country} could not be fetched.",
                "data": {"error": str(e)},
                "sources": [{"name": "disease.sh (JHU)", "url": "https://disease.sh/"}],
            }

    def fetch_medicine_info(self, medicine_name, info_type="general"):
        if not medicine_name:
            return {
                "type": "medicine_error",
                "summary": "No medicine specified.",
                "data": {},
                "sources": [],
            }
        try:
            med_query = re.sub(r"\s+", "+", medicine_name)
            url = self.medicine_api + med_query + "&limit=20"
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            data = response.json()

            reactions = []
            for result in data.get("results", []):
                for r in result.get("patient", {}).get("reaction", []):
                    name = r.get("reactionmeddrapt")
                    if name:
                        reactions.append(name.strip())

            from collections import Counter
            counts = Counter([x.lower() for x in reactions])
            top = [name for name, _ in counts.most_common(6)]
            pretty = ", ".join(top) if top else "No common side effects found."

            return {
                "type": "medicine_side_effects",
                "summary": f"Commonly reported side effects for {medicine_name}: {pretty}.",
                "data": {"medicine": medicine_name, "side_effects": sorted(set(reactions))},
                "sources": [{"name": "FDA (openFDA)", "url": "https://open.fda.gov/apis/drug/event/"}],
            }

        except Exception as e:
            return {
                "type": "medicine_error",
                "summary": f"Could not fetch medicine info for {medicine_name}.",
                "data": {"error": str(e)},
                "sources": [{"name": "FDA (openFDA)", "url": "https://open.fda.gov/apis/drug/event/"}],
            }

    def fetch_nutrition_info(self, query_text: str):
        """
        Look up foods from USDA FoodData Central and build a friendly summary that
        lists the top 3 foods for several key nutrients (when available).
        Returns:
        {
            "type": "nutrition",
            "summary": <string>,                  # human-friendly lines
            "results": [ {food_name, nutrients} ],
            "summary_table": [
            {
                "Nutrient": <str>,
                "Top Foods": [ {"food_name": <str>, "amount": <float>} , ... up to 3 ]
            },
            ...
            ]
        }
        On failure:
        {
            "type": "nutrition_error",
            "summary": "USDA query failed (check key/limit/network).",
            "error": <str>
        }
        """
        try:
            # --- 1) Primary query ---
            params = {"query": query_text, "api_key": self.usda_api_key, "pageSize": 8}
            response = requests.get(self.usda_api, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            foods = data.get("foods", []) or []

            # --- 1b) Simple fallback keyword if no hits ---
            if not foods:
                qlow = (query_text or "").lower()
                fallback = None
                if "vitamin c" in qlow or "ascorbic" in qlow:
                    fallback = "vitamin c"
                elif "protein" in qlow:
                    fallback = "protein"
                elif "iron" in qlow:
                    fallback = "iron"
                elif "calcium" in qlow:
                    fallback = "calcium"

                if fallback:
                    response = requests.get(
                        self.usda_api,
                        params={"query": fallback, "api_key": self.usda_api_key, "pageSize": 8},
                        timeout=20,
                    )
                    response.raise_for_status()
                    data = response.json()
                    foods = data.get("foods", []) or []

            # --- 2) Normalize results: [{food_name, nutrients{...}}] ---
            results = []
            for item in foods:
                food_name = item.get("description")
                nutrients = {
                    n.get("nutrientName"): n.get("value")
                    for n in (item.get("foodNutrients") or [])
                    if n and n.get("nutrientName") is not None
                }
                results.append({"food_name": food_name, "nutrients": nutrients})

            if not results:
                return {
                    "type": "nutrition",
                    "summary": "No USDA foods matched that query.",
                    "results": [],
                    "summary_table": [],
                }

            # --- 3) Build top-3 list for each key nutrient ---
            key_nutrients = [
                "Vitamin C, total ascorbic acid",
                "Protein",
                "Calcium, Ca",
                "Iron, Fe",
                "Vitamin A, RAE",
            ]

            def _fmt_amount(v):
                try:
                    # keep small decimals readable, big numbers as ints
                    fv = float(v)
                    return int(fv) if abs(fv - int(fv)) < 1e-6 else round(fv, 2)
                except Exception:
                    return v

            summary_table = []
            for nutrient in key_nutrients:
                foods_with_nutrient = [
                    {
                        "food_name": r["food_name"],
                        "amount": r["nutrients"].get(nutrient, 0) or 0,
                    }
                    for r in results
                ]
                # Only keep > 0 values
                foods_with_nutrient = [f for f in foods_with_nutrient if (f["amount"] or 0) > 0]
                foods_with_nutrient.sort(key=lambda x: x["amount"], reverse=True)

                if foods_with_nutrient:
                    top_foods = [
                        {"food_name": f["food_name"], "amount": _fmt_amount(f["amount"])}
                        for f in foods_with_nutrient[:3]
                    ]
                    summary_table.append({"Nutrient": nutrient, "Top Foods": top_foods})

            # --- 4) Human-friendly multi-line summary ---
            if summary_table:
                lines = []
                for row in summary_table:
                    foods_str = ", ".join(
                        f"{f['food_name']} ({f['amount']})" for f in row["Top Foods"]
                    )
                    lines.append(f"{row['Nutrient']}: {foods_str}")
                facts_summary = "Top nutrient-rich foods:\n" + "\n".join(lines)
            else:
                facts_summary = "USDA results found, but no notable amounts for the selected nutrients."

            return {
                "type": "nutrition",
                "summary": facts_summary,
                "results": results,
                "summary_table": summary_table,
            }

        except Exception as e:
            return {
                "type": "nutrition_error",
                "summary": "USDA query failed (check key/limit/network).",
                "error": str(e),
            }


    # Keep the rest of your pipeline the same, but pass these fields through
    # so adapters (covid/dengue/malaria/worldbank) can use them.

    # ----------------- Routing -----------------
    def search(self, question: str) -> Dict[str, Any]:
        """
        Returns a uniform envelope for the orchestrator/front-end:
        {
            "type": "retrieval",
            "query": <parsed query>,
            "facts": <normalized payload from adapter/builtin>,
            "summary": <string>,
            "sources": [ {name, url}, ... ]
        }
        """
        query = self.extract_keywords(question)

        # 1) Try adapters first
        for adapter in self.adapters:
            try:
                if adapter.supports(query):
                    result = adapter.fetch(query)  # {type, summary, data, sources?}
                    return {
                        "type": "retrieval",
                        "query": query,
                        "facts": result,
                        "summary": result.get("summary", ""),
                        "sources": result.get("sources", []),
                    }
            except Exception as e:
                return {
                    "type": "retrieval",
                    "query": query,
                    "facts": {"type": "adapter_error", "data": {"error": str(e)}},
                    "summary": f"{adapter.__class__.__name__} failed to fetch data.",
                    "sources": [],
                }

        # 2) Built-ins
        if query.get("disease") == "covid":
            payload = self.fetch_covid_data(query.get("country") or "World", query.get("info_type"))
            return {
                "type": "retrieval",
                "query": query,
                "facts": payload,
                "summary": payload.get("summary", ""),
                "sources": payload.get("sources", []),
            }

        if query.get("medicine"):
            payload = self.fetch_medicine_info(query["medicine"], query.get("info_type"))
            return {
                "type": "retrieval",
                "query": query,
                "facts": payload,
                "summary": payload.get("summary", ""),
                "sources": payload.get("sources", []),
            }

        if query.get("info_type") == "nutrition":
            payload = self.fetch_nutrition_info(question)
            return {
                "type": "retrieval",
                "query": query,
                "facts": payload,
                "summary": payload.get("summary", ""),
                "sources": payload.get("sources", []),
            }

        # 3) Fallback
        return {
            "type": "retrieval",
            "query": query,
            "facts": {"type": "general_health", "data": {"info": "No structured adapter matched."}},
            "summary": "I couldn't match that to a real-world data source, but I can try a general answer.",
            "sources": [],
        }

    # --- paste these inside the InformationRetrievalAgent class ---

    def web_search(self, question: str, filters: dict | None = None) -> dict:
        """
        Deterministic 'links-only' search:
        - Parse disease/country from our existing extractor
        - Return a small set of authoritative links (title, url, source)
        """
        q = (question or "").strip()
        parsed = self.extract_keywords(q)
        disease = (parsed.get("disease") or "").lower()
        country = parsed.get("country") or parsed.get("region") or "World"

        # helper lookups (if you already have these helpers, reuse them;
        # otherwise you can no-op them: iso2='world', slug='world')
        try:
            iso2 = _iso2_or_none(country) or "world"   # e.g., 'us' for United States
        except NameError:
            iso2 = "world"
        try:
            slug = _owid_slug(country)                 # e.g., 'united-states', 'sri-lanka'
        except NameError:
            slug = "world"

        items: list[dict] = []

        # --- COVID-19 ---
        if "covid" in disease:
            # Our World in Data (country or global)
            if slug and slug != "world":
                items.append({
                    "title": f"COVID-19 data for {country} – Our World in Data",
                    "url": f"https://ourworldindata.org/coronavirus/country/{slug}",
                    "source": "Our World in Data",
                })
            else:
                items.append({
                    "title": "Global COVID-19 data – Our World in Data",
                    "url": "https://ourworldindata.org/coronavirus",
                    "source": "Our World in Data",
                })

            items.append({
                "title": "WHO COVID-19 Global Dashboard",
                "url": "https://covid19.who.int/",
                "source": "WHO",
            })
            items.append({
                "title": "disease.sh – COVID-19 (JHU) API",
                "url": "https://disease.sh/docs/",
                "source": "disease.sh",
            })
            if iso2 == "us":
                items.append({
                    "title": "CDC COVID Data Tracker (US)",
                    "url": "https://covid.cdc.gov/covid-data-tracker/",
                    "source": "CDC",
                })

        # --- Dengue ---
        elif "dengue" in disease:
            items.append({
                "title": "WHO – Dengue and severe dengue (fact sheet)",
                "url": "https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue",
                "source": "WHO",
            })
            items.append({
                "title": "PAHO/WHO – Dengue situation (Region of the Americas)",
                "url": "https://www.paho.org/en/topics/dengue",
                "source": "PAHO/WHO",
            })
            if slug and slug != "world":
                items.append({
                    "title": f"WHO Country profile – {country}",
                    "url": f"https://www.who.int/countries/{slug}",
                    "source": "WHO",
                })

        # --- Malaria ---
        elif "malaria" in disease:
            items.append({
                "title": "WHO – Global Malaria Programme",
                "url": "https://www.who.int/teams/global-malaria-programme",
                "source": "WHO",
            })
            items.append({
                "title": "World Malaria Report – WHO",
                "url": "https://www.who.int/teams/global-malaria-programme/reports/world-malaria-report",
                "source": "WHO",
            })
            if slug and slug != "world":
                items.append({
                    "title": f"WHO Country profile – {country}",
                    "url": f"https://www.who.int/countries/{slug}",
                    "source": "WHO",
                })

        # --- Influenza / Flu ---
        elif disease in ("flu", "influenza"):
            items.append({
                "title": "WHO – Influenza (seasonal)",
                "url": "https://www.who.int/health-topics/influenza-seasonal",
                "source": "WHO",
            })
            items.append({
                "title": "CDC – FluView (US influenza surveillance)",
                "url": "https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html",
                "source": "CDC",
            })

        # --- Tuberculosis ---
        elif "tb" in disease or "tuberculosis" in disease:
            items.append({
                "title": "WHO – Global Tuberculosis Programme",
                "url": "https://www.who.int/teams/global-tuberculosis-programme",
                "source": "WHO",
            })
            items.append({
                "title": "WHO – Global TB Report",
                "url": "https://www.who.int/teams/global-tuberculosis-programme/tb-reports",
                "source": "WHO",
            })

        # --- HIV ---
        elif disease == "hiv":
            items.append({
                "title": "WHO – HIV/AIDS",
                "url": "https://www.who.int/health-topics/hiv-aids",
                "source": "WHO",
            })
            items.append({
                "title": "UNAIDS – Data",
                "url": "https://www.unaids.org/en/resources/documents",
                "source": "UNAIDS",
            })

        # --- Measles ---
        elif "measles" in disease:
            items.append({
                "title": "WHO – Measles",
                "url": "https://www.who.int/health-topics/measles",
                "source": "WHO",
            })
            items.append({
                "title": "CDC – Measles (US)",
                "url": "https://www.cdc.gov/measles/",
                "source": "CDC",
            })

        # --- Fallback general portals ---
        else:
            items.extend([
                {
                    "title": "WHO – Data",
                    "url": "https://www.who.int/data",
                    "source": "WHO",
                },
                {
                    "title": "Our World in Data – Health",
                    "url": "https://ourworldindata.org/health-meta",
                    "source": "Our World in Data",
                },
                {
                    "title": "World Bank – Health Nutrition and Population Data",
                    "url": "https://datatopics.worldbank.org/health/",
                    "source": "World Bank",
                },
            ])

        return {
            "type": "search",
            "query": {"raw": q, **parsed},
            "items": items[:10],
            "sources": [],
        }

    def web_search_links_only(self, question: str, filters: dict | None = None) -> dict:
        """
        Wrap web_search but expose only the links array, to keep the payload minimal.
        """
        full = self.web_search(question, filters or {})
        return {
            "type": "search",
            "items": full.get("items", []),
            "query": full.get("query", {}),
            "sources": [],
        }


    def suggest(self, q: str) -> Dict[str, Any]:
        """
        Very simple suggester (you can enhance later by edge n-grams or static dictionaries).
        Returns up to ~8 suggestions relevant to health topics.
        """
        base = [
            "vitamin a", "vitamin b12", "vitamin c", "vitamin d",
            "iron deficiency", "calcium rich foods", "protein intake",
            "covid cases", "covid deaths", "influenza trend",
            "life expectancy", "under 5 mortality", "health expenditure",
            "dengue symptoms", "malaria prevention", "tb incidence",
        ]
        ql = (q or "").strip().lower()
        if not ql:
            return {"suggestions": base[:8]}

        out = [s for s in base if ql in s.lower()]
        # If nothing matches, still return a couple of helpful ideas
        if not out:
            out = base[:5]
        return {"suggestions": out[:8]}


    
 