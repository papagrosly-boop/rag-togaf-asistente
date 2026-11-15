"""Prueba las tools SIN el LLM con .invoke() (captura para el PDF, sección 1.2)."""
from tools import buscar_en_base, buscar_web, consultar_adm, generar_referencia_apa

print("=== buscar_en_base.invoke ===")
print(buscar_en_base.invoke({"consulta": "objectives of the Preliminary Phase"}))
print("\n=== consultar_adm.invoke ===")
print(consultar_adm.invoke({"fase": "C", "operacion": "siguiente"}))
print(consultar_adm.invoke({"fase": "B", "operacion": "faltantes"}))
print("\n=== generar_referencia_apa.invoke ===")
print(generar_referencia_apa.invoke({"documento": "Fase Preliminar"}))
print("\n=== buscar_web.invoke (solo sitios permitidos) ===")
print(buscar_web.invoke({"consulta": "TOGAF 9.1 Phase E Opportunities and Solutions objectives"}))
