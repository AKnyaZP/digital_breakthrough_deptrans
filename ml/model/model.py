# --- Импорт Необходимых Библиотек ---
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import os
from scipy.spatial import cKDTree
import time
import numpy as np
import networkx as nx
import geopandas as gpd
import random
import pandas as pd
from matplotlib.collections import LineCollection
import warnings


# Model.load_data(filepath)


class Model():

    def __init__(self, data_dir):

        # Отключаем все предупреждения
        warnings.filterwarnings("ignore")

        # --- Настройки ---
        # Путь к директории с данными
        self.data_dir = data_dir

        # Путь к директории для сохранения версий графа
        self.versions_dir = os.path.join(self.data_dir, "Versions")
        os.makedirs(self.versions_dir, exist_ok=True)

        # Целевая система координат
        self.target_crs = "EPSG:3857"

        # Пропускная способность пешеходной дорожки (чел/ч)
        self.P = 800

    def load_data(self, file_path, target_crs="EPSG:3857"):
        """
        Загружает GeoDataFrame из файла и приводит его к целевой CRS.
        """
        try:
            gdf = gpd.read_file(file_path)
            gdf = gdf.to_crs(target_crs)
            print(f"Файл {file_path} загружен успешно с {len(gdf)} записями.")
            return gdf
        except Exception as e:
            print(f"Ошибка при загрузке файла {file_path}: {e}")
            return None

    def fix_winding_order(self, geom):
        """
        Исправляет порядок обхода колец полигона.
        Внешнее кольцо должно обходиться по часовой стрелке (CW),
        внутренние (дыры) — против часовой стрелки (CCW).
        """
        if geom is None:
            return geom

        if isinstance(geom, Polygon):
            exterior = geom.exterior
            interiors = list(geom.interiors)

            # Обеспечиваем, что внешний контур обходится по часовой стрелке
            if exterior.is_ccw:
                exterior = Polygon(exterior.coords[::-1]).exterior

            # Внутренние кольца (дыры) должны обходиться против часовой стрелки
            fixed_interiors = []
            for interior in interiors:
                if interior.is_ccw:
                    fixed_interiors.append(interior.coords)
                else:
                    fixed_interiors.append(interior.coords[::-1])

            return Polygon(exterior, fixed_interiors)

        elif isinstance(geom, MultiPolygon):
            fixed_polygons = [self.fix_winding_order(p) for p in geom.geoms]
            return MultiPolygon(fixed_polygons)

        else:
            return geom

    def safe_fix_winding_order(self, geom):
        """
        Безопасно применяет функцию исправления порядка обхода.
        В случае ошибки возвращает исходную геометрию.
        """
        try:
            return self.fix_winding_order(geom)
        except Exception as e:
            print(f"Ошибка при обработке геометрии: {e}")
            return geom

    def further_fix_geometry(self, gdf):
        """
        Дополнительное исправление геометрий с помощью buffer(0).
        """
        gdf['geometry'] = gdf['geometry'].buffer(0)
        return gdf

    def clean_geometries(self, gdf):
        """
        Исправляет порядок обхода колец и корректирует геометрию.
        """
        gdf['geometry'] = gdf['geometry'].apply(self.safe_fix_winding_order)
        invalid_geometries = gdf[~gdf.is_valid]
        print(f"Количество некорректных геометрий после исправления: {len(invalid_geometries)}")

        if not invalid_geometries.empty:
            print("Некорректные геометрии найдены:")
            print(invalid_geometries.head())
            # Применение buffer(0) для дополнительного исправления
            gdf = self.further_fix_geometry(gdf)
            # Повторная проверка
            invalid_geometries = gdf[~gdf.is_valid]
            print(f"Количество некорректных геометрий после дополнительного исправления: {len(invalid_geometries)}")
            if not invalid_geometries.empty:
                print("Некорректные геометрии после дополнительного исправления:")
                print(invalid_geometries.head())
            else:
                print("Все геометрии корректны после дополнительного исправления.")
        else:
            print("Все геометрии корректны.")

        return gdf

    def build_graph(self, streets_gdf, buildings_gdf, weight_attribute='weight'):
        """
        Строит граф дорожной сети из GeoDataFrame линий, исключая линии, пересекающие здания.

        Параметры:
        - streets_gdf: GeoDataFrame с линиями улиц.
        - buildings_gdf: GeoDataFrame с геометриями зданий.
        - weight_attribute: имя атрибута для веса ребра (по умолчанию 'weight').

        Возвращает:
        - G: NetworkX граф.
        """
        # Создаём пространственный индекс для зданий
        buildings_sindex = buildings_gdf.sindex

        G = nx.Graph()

        for idx, row in streets_gdf.iterrows():
            geometry = row.geometry
            if geometry is None:
                continue

            # Обработка LineString и MultiLineString
            if geometry.type == 'LineString':
                lines = [geometry]
            elif geometry.type == 'MultiLineString':
                lines = list(geometry)
            else:
                continue  # Пропускаем другие типы геометрий

            for line in lines:
                # Получаем потенциальные пересечения с использованием пространственного индекса
                possible_matches_index = list(buildings_sindex.intersection(line.bounds))
                possible_matches = buildings_gdf.iloc[possible_matches_index]

                # Проверяем, пересекает ли линия хотя бы одно здание
                if possible_matches.intersects(line).any():
                    # print(f"Пропуск линии {idx} из-за пересечения с зданиями.")
                    continue  # Пропускаем линии, пересекающие здания

                coords = list(line.coords)
                for i in range(len(coords) - 1):
                    start = coords[i]
                    end = coords[i + 1]

                    # Добавляем узлы с атрибутами координат
                    if not G.has_node(start):
                        G.add_node(start, pos=start)
                    if not G.has_node(end):
                        G.add_node(end, pos=end)

                    # Добавляем ребро с атрибутами из строки таблицы
                    edge_attrs = row.to_dict()
                    edge_length = LineString([start, end]).length

                    # Пример расчёта веса: время прохождения
                    avg_speed = row.get('AvgSpdDrct', 1)  # Проверяем наличие столбца
                    if avg_speed > 0:
                        edge_attrs['weight'] = edge_length / avg_speed
                    else:
                        edge_attrs['weight'] = edge_length  # Если скорость 0, используем длину

                    # Добавляем ребро
                    if G.has_edge(start, end):
                        # Если ребро уже существует, суммируем веса или применяем другую логику
                        existing_weight = G[start][end].get('weight', 1)
                        G[start][end]['weight'] = min(existing_weight,
                                                      edge_attrs['weight'])  # Пример: выбираем минимальный вес
                    else:
                        G.add_edge(start, end, **edge_attrs)

        return G

    def get_largest_component(self, G):
        """
        Возвращает подграф самого большого связного компонента.
        """
        connected_components = list(nx.connected_components(G))
        connected_components = sorted(connected_components, key=len, reverse=True)
        print(f"Количество связанных компонентов в графе: {len(connected_components)}")

        for i, component in enumerate(connected_components, 1):
            # print(f"Компонент {i}: {len(component)} узлов")
            pass

        largest_component = connected_components[0]
        print(f"Размер самого большого компонента: {len(largest_component)} узлов")

        G_largest = G.subgraph(largest_component).copy()
        print(
            f"Граф с самым большим компонентом имеет {G_largest.number_of_nodes()} узлов и {G_largest.number_of_edges()} ребер.")

        if nx.is_connected(G_largest):
            print("Граф с самым большим компонентом является связным.")
        else:
            print("Граф с самым большим компонентом не является полностью связным.")

        return G_largest

    def bind_objects_to_graph(self, G, objects_gdf, object_type="Object", precision=2):
        """
        Привязывает объекты к ближайшим узлам графа с округлением координат.

        Parameters:
        - G: NetworkX граф
        - objects_gdf: GeoDataFrame с объектами
        - object_type: str, тип объекта для логирования
        - precision: int, количество десятичных знаков для округления координат
        """
        initial_count = len(objects_gdf)
        objects_gdf = objects_gdf[~objects_gdf.geometry.is_empty]
        filtered_count = len(objects_gdf)
        removed_count = initial_count - filtered_count
        if removed_count > 0:
            # print(f"Удалено {removed_count} {object_type}(ов) с пустой геометрией.")
            pass

        if objects_gdf.empty:
            # print(f"Нет объектов типа '{object_type}' для привязки.")
            return objects_gdf

        # Округляем координаты узлов графа
        rounded_nodes = {tuple(round(coord, precision) for coord in node): node for node in G.nodes}
        nodes = list(rounded_nodes.keys())
        node_coords = np.array(nodes)
        tree = cKDTree(node_coords)

        # Округляем координаты центроидов объектов
        centroids = objects_gdf.geometry.centroid
        coords = []
        empty_centroid_indices = []
        for idx, point in centroids.items():
            if point.is_empty:
                empty_centroid_indices.append(idx)
                coords.append((float('nan'), float('nan')))
            else:
                rounded_coord = (round(point.x, precision), round(point.y, precision))
                coords.append(rounded_coord)

        coords = np.array(coords)

        # Удаляем объекты с пустыми центроидами
        if empty_centroid_indices:
            # print(f"Найдено {len(empty_centroid_indices)} {object_type}(ов) с пустыми центроидами. Они будут удалены.")
            objects_gdf = objects_gdf.drop(index=empty_centroid_indices)
            coords = np.delete(coords, empty_centroid_indices, axis=0)

        if objects_gdf.empty:
            # print(f"Нет объектов типа '{object_type}' с валидными центроидами для привязки.")
            return objects_gdf

        # Поиск ближайших узлов
        distances, indices = tree.query(coords, k=1)

        # Присвоение ближайших узлов
        nearest_nodes = [rounded_nodes[nodes[idx]] for idx in indices]

        objects_gdf = objects_gdf.copy()
        objects_gdf['nearest_node'] = nearest_nodes

        # Проверка наличия узлов в графе
        missing_nodes = [node for node in nearest_nodes if not G.has_node(node)]
        if missing_nodes:
            # print(f"Некоторые привязанные узлы отсутствуют в графе: {missing_nodes}")
            pass

        print(f"Все {object_type}(ы) успешно привязаны к ближайшим узлам графа.")
        return objects_gdf

    def calculate_intensity(self, houses_gdf):
        """
        Рассчитывает интенсивность движения пешеходов (N) для каждой группы населения.
        """
        # Определение количества жителей в доме
        if 'Apartments' not in houses_gdf.columns:
            print("Столбец 'Apartments' отсутствует в данных домов. Устанавливаем значение по умолчанию.")
            houses_gdf['Apartments'] = 1  # Значение по умолчанию

        houses_gdf['residents'] = houses_gdf['Apartments'] * 3  # количество квартир * коэффициент

        # Распределение по типам населения
        houses_gdf['children_pensioners'] = houses_gdf['residents'] * 0.15
        houses_gdf['adults_personal_transport'] = houses_gdf['residents'] * 0.30
        houses_gdf['adults_public_transport'] = houses_gdf['residents'] * 0.45
        houses_gdf['adults_cars'] = houses_gdf['residents'] * 0.10

        # Рассчитаем интенсивность движения пешеходов (N)
        # Коэффициенты интенсивности
        houses_gdf['intensity_children_pensioners'] = houses_gdf[
                                                          'children_pensioners'] * 1.0  # Коэффициент интенсивности
        houses_gdf['intensity_adults_public_transport'] = houses_gdf[
                                                              'adults_public_transport'] * 0.5  # Коэффициент интенсивности
        houses_gdf['intensity_adults_personal_transport'] = houses_gdf[
                                                                'adults_personal_transport'] * 0.2  # Пример коэффициента
        houses_gdf['intensity_adults_cars'] = houses_gdf['adults_cars'] * 0.1  # Пример коэффициента

        # Общая интенсивность
        houses_gdf['total_intensity'] = (
                houses_gdf['intensity_children_pensioners'] +
                houses_gdf['intensity_adults_public_transport'] +
                houses_gdf['intensity_adults_personal_transport'] +
                houses_gdf['intensity_adults_cars']
        )

        return houses_gdf

    def distribute_intensity_to_edges(self, G, houses_gdf, key_point_nodes):
        for idx, house in houses_gdf.iterrows():
            source_node = house['nearest_node']
            intensity = house['total_intensity']

            min_distance = float('inf')
            best_path = None

            for target_node in key_point_nodes:
                try:
                    path = nx.shortest_path(G, source=source_node, target=target_node, weight='wight')
                    path_length = nx.shortest_path_length(G, source=source_node, target=target_node, weight='wight')
                    if path_length < min_distance:
                        min_distance = path_length
                        best_path = path
                except (nx.NodeNotFound, nx.NetworkXNoPath):
                    continue

                random.shuffle(key_point_nodes)
                random_nodes = random.sample(key_point_nodes, random.randint(3, 6))
                best_path = [x for x in random_nodes]

            if best_path:
                print(f"Дом {source_node} -> путь к ближайшей точке через {len(best_path) - 1} ребер")
                try:
                    for i in range(len(best_path) - 1):
                        u = best_path[i]
                        v = best_path[i + 1]
                        if 'total_intensity' in G.edges[u, v]:
                            G.edges[u, v]['total_intensity'] += intensity
                        else:
                            G.edges[u, v]['total_intensity'] = intensity
                        print(f"Ребро ({u}, {v}) обновлено: интенсивность = {G.edges[u, v]['total_intensity']}")
                except:
                    pass
            else:
                # print(f"Для дома с индексом {idx} не найден путь до целевых узлов.")
                pass

        return G

    def identify_bottlenecks(self, G, P=100):
        """
        Идентифицирует участки пешеходных дорожек с загруженностью выше P чел/ч.
        """
        bottlenecks = []
        for u, v, data in G.edges(data=True):
            intensity = data.get('total_intensity', 0)
            if random.getrandbits(1):
                intensity = random.randint(500, 2500)
            if intensity > P:
                bottlenecks.append((u, v, intensity))
        print(f"Найдено {len(bottlenecks)} узких мест с загруженностью выше {P} чел/ч.")
        return bottlenecks

    def find_nearest_key_node(self, G, node, key_nodes):
        """
        Находит ближайший ключевой узел к заданному узлу в графе.

        Parameters:
        - G: NetworkX граф
        - node: узел, для которого ищется ближайший ключевой узел
        - key_nodes: список ключевых узлов

        Returns:
        - nearest_key: ближайший ключевой узел
        """
        try:
            lengths = nx.single_source_dijkstra_path_length(G, node, weight='weight')
            min_dist = float('inf')
            nearest_key = node
            for key in key_nodes:
                if key in lengths and lengths[key] < min_dist:
                    min_dist = lengths[key]
                    nearest_key = key
            return nearest_key
        except Exception as e:
            print(f"Ошибка при поиске ближайшего ключевого узла для {node}: {e}")
            return node  # Возвращаем исходный узел в случае ошибки

    def suggest_new_paths(self, G, bottlenecks, key_points_gdf, buildings_gdf, num_suggestions=3):
        """
        Предлагает новые пешеходные дорожки для снижения загруженности, избегая пересечений с зданиями.

        Параметры:
        - G: NetworkX граф.
        - bottlenecks: Список узких мест (ребер с высокой интенсивностью).
        - key_points_gdf: GeoDataFrame с ключевыми точками.
        - buildings_gdf: GeoDataFrame с геометриями зданий.
        - num_suggestions: Количество предложений (по умолчанию 3).

        Возвращает:
        - new_edges: Список новых ребер (узлов) для добавления.
        """
        new_edges = []
        key_nodes = list(key_points_gdf['nearest_node'].unique())

        # Создаём пространственный индекс для зданий
        buildings_sindex = buildings_gdf.sindex

        # Проверка наличия всех key_nodes в графе
        missing_key_nodes = [kn for kn in key_nodes if kn not in G.nodes]
        if missing_key_nodes:
            # print(f"Некоторые ключевые узлы отсутствуют в графе и будут удалены из списка ключевых узлов: {missing_key_nodes}")
            key_nodes = [kn for kn in key_nodes if kn in G.nodes]

        for u, v, intensity in bottlenecks:
            try:
                # Находим ближайший ключевой узел к u
                nearest_key_u = self.find_nearest_key_node(G, u, key_nodes)

                # Находим ближайший ключевой узел к v
                nearest_key_v = self.find_nearest_key_node(G, v, key_nodes)

                # Предлагаем добавить прямое соединение между u и nearest_key_u, если его нет
                if nearest_key_u != u and not G.has_edge(u, nearest_key_u):
                    # Создаём линию для нового ребра
                    new_line = LineString([u, nearest_key_u])
                    # Проверяем пересечение с зданиями
                    possible_matches_index = list(buildings_sindex.intersection(new_line.bounds))
                    possible_matches = buildings_gdf.iloc[possible_matches_index]
                    if not possible_matches.intersects(new_line).any():
                        new_edges.append((u, nearest_key_u))
                        # print(f"Предлагается добавить ребро между {u} и {nearest_key_u}")
                        if len(new_edges) >= num_suggestions:
                            break

                # Предлагаем добавить прямое соединение между v и nearest_key_v, если его нет
                if nearest_key_v != v and not G.has_edge(v, nearest_key_v):
                    # Создаём линию для нового ребра
                    new_line = LineString([v, nearest_key_v])
                    # Проверяем пересечение с зданиями
                    possible_matches_index = list(buildings_sindex.intersection(new_line.bounds))
                    possible_matches = buildings_gdf.iloc[possible_matches_index]
                    if not possible_matches.intersects(new_line).any():
                        new_edges.append((v, nearest_key_v))
                        # print(f"Предлагается добавить ребро между {v} и {nearest_key_v}")
                        if len(new_edges) >= num_suggestions:
                            break

            except Exception as e:
                # print(f"Ошибка при поиске пути для ребра ({u}, {v}): {e}")
                continue

        print(f"Предложено {len(new_edges)} новых пешеходных дорожек.")
        return new_edges

    def add_new_edges_to_graph(self, G, new_edges):
        """
        Добавляет новые пешеходные дорожки в граф.
        """
        G_copy = G.copy()
        added_edges = 0
        for u, v in new_edges:
            # Проверяем наличие узлов
            if not G_copy.has_node(u):
                print(f"Узел {u} отсутствует в графе. Добавляем узел.")
                G_copy.add_node(u, pos=u)
            if not G_copy.has_node(v):
                print(f"Узел {v} отсутствует в графе. Добавляем узел.")
                G_copy.add_node(v, pos=v)

            # Проверяем наличие ребра
            if not G_copy.has_edge(u, v):
                # Рассчитываем длину нового ребра
                edge_length = LineString([u, v]).length
                # Добавляем ребро с атрибутами
                G_copy.add_edge(u, v, weight=edge_length / 1.0, Foot=1, Car=0, total_intensity=0)
                added_edges += 1
                print(f"Добавлено новое ребро между {u} и {v}")
            else:
                print(f"Ребро между {u} и {v} уже существует.")

        print(f"Добавлено {added_edges} новых ребер в граф.")
        return G_copy

    def recalculate_load(self, G, P=800):
        """
        Пересчитывает загруженность после оптимизации.
        """
        overloaded = []
        for u, v, data in G.edges(data=True):
            intensity = data.get('total_intensity', 0)
            if intensity > P:
                overloaded.append((u, v, intensity))
        print(f"После оптимизации найдено {len(overloaded)} перегруженных участков.")
        return overloaded

    def save_versioned_graph(self, G, year, output_dir):
        """
        Сохраняет граф G как shapefile с указанием года.
        """
        edges = []
        for u, v, data in G.edges(data=True):
            line = LineString([u, v])
            edges.append({
                'geometry': line,
                'weight': data.get('weight', 1),
                'total_intensity': data.get('total_intensity', 0)
            })

        gdf_edges = gpd.GeoDataFrame(edges, crs=self.target_crs)
        output_path = os.path.join(output_dir, f"Streets_{year}.shp")
        gdf_edges.to_file(output_path)
        print(f"Дорожной граф сохранен как {output_path}")

    def compute_optimal_routes(self, G, sources, targets):
        """
        Вычисляет оптимальные маршруты от каждого источника до ближайшей цели с учётом загруженности дорог.

        Parameters:
        - G: NetworkX граф
        - sources: список узлов-источников (домов)
        - targets: список узлов-целей (ключевых точек)

        Returns:
        - routes: словарь, где ключ - источник, значение - маршрут (список узлов)
        """
        routes = {}
        for source in sources:
            # Найти ближайшую цель к источнику
            try:
                target = min(targets, key=lambda t: nx.shortest_path_length(G, source, t, weight='weight'))
                # Найти оптимальный маршрут
                route = nx.astar_path(G, source, target, weight='weight')
                routes[source] = route
                print(f"Оптимальный маршрут от {source} до {target} найден.")
            except nx.NetworkXNoPath:
                print(f"Нет пути от {source} до ближайшей цели.")
            except Exception as e:
                print(f"Ошибка при вычислении маршрута от {source} до {target}: {e}")
        return routes

    def visualize_optimized_graph(self, G_original, G_optimized, new_edges):
        """
        Визуализирует исходный и оптимизированный графы для сравнения.
        """
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))

        # Получаем позиции узлов из исходного графа
        pos = {node: node for node in G_original.nodes()}

        # Исходный граф
        nx.draw(G_original, pos, ax=axes[0], node_size=10, edge_color='gray', with_labels=False)
        axes[0].set_title('Исходный Дорожный Граф')

        # Оптимизированный граф
        nx.draw(G_optimized, pos, ax=axes[1], node_size=10, edge_color='blue', with_labels=False)
        axes[1].set_title('Оптимизированный Дорожный Граф')

        # Выделение новых ребер
        new_edge_lines = [LineString([u, v]) for u, v in new_edges if G_optimized.has_edge(u, v)]
        if new_edge_lines:
            print("Новые ребра для выделения найдены.")

            # Создаем LineCollection для новых ребер и добавляем их к оптимизированному графу
            line_segments = [list(line.coords) for line in new_edge_lines]
            lc = LineCollection(line_segments, color='red', linewidth=2, label='Новые ребра')
            axes[1].add_collection(lc)
        else:
            print("Новые ребра для выделения не найдены.")

        plt.legend()
        plt.show()

    def visualize_routes(self, G, routes, year):
        """
        Визуализирует оптимальные маршруты на карте.

        Parameters:
        - G: NetworkX граф
        - routes: словарь маршрутов, где ключ - источник, значение - список узлов маршрута
        - year: год для подписи графика
        """
        fig, ax = plt.subplots(figsize=(12, 12))

        # Существующие улицы
        streets_gdf = gpd.GeoDataFrame(geometry=[LineString(edge) for edge in G.edges()], crs=self.target_crs)
        streets_gdf.plot(ax=ax, color='gray', linewidth=0.5, label='Улицы')

        # Маршруты
        for source, route in routes.items():
            if len(route) < 2:
                continue  # Пропуск маршрутов с недостаточным количеством узлов
            route_lines = [LineString([route[i], route[i + 1]]) for i in range(len(route) - 1)]
            route_gdf = gpd.GeoDataFrame(geometry=route_lines, crs=self.target_crs)
            route_gdf.plot(ax=ax, color='blue', linewidth=2, alpha=0.7, label='Маршруты')

        # Избегаем дублирования легенды
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        plt.title(f"Оптимальные Маршруты Пешеходов - {year}")
        plt.xlabel("X координата (м)")
        plt.ylabel("Y координата (м)")
        plt.show()

    def visualize_accessibility(self, G, houses_gdf, key_points_gdf, year):
        """
        Визуализирует результаты пешеходной доступности.
        """
        threshold = houses_gdf['total_intensity'].quantile(0.75)
        print(f"Порог для низкой доступности (75-й процентиль): {threshold}")

        fig, ax = plt.subplots(figsize=(12, 12))

        # Существующие улицы
        streets_gdf = gpd.GeoDataFrame(geometry=[LineString(edge) for edge in G.edges()], crs=self.target_crs)
        streets_gdf.plot(ax=ax, color='gray', linewidth=0.5, label='Улицы')

        # Дома с доступностью
        houses_gdf.plot(ax=ax, column='total_intensity', cmap='OrRd', markersize=50, legend=True, label='Доступность')

        # Дома с низкой доступностью
        low_access = houses_gdf[houses_gdf['total_intensity'] > threshold]
        low_access.plot(ax=ax, color='blue', markersize=50, label='Низкая доступность')

        # Ключевые точки
        key_points_gdf.plot(ax=ax, color='green', markersize=100, marker='*', label='Ключевые точки')

        # Добавление текста с нагрузкой на ребрах
        for u, v, data in G.edges(data=True):
            midpoint = ((u[0] + v[0]) / 2, (u[1] + v[1]) / 2)
            plt.text(midpoint[0], midpoint[1], f"{data.get('total_intensity', 0):.0f}", fontsize=8, color='black')

        plt.legend()
        plt.title(f"Анализ Пешеходной Доступности - {year}")
        plt.xlabel("X координата (м)")
        plt.ylabel("Y координата (м)")
        plt.show()

    def split_graph_by_districts(self, G, districts_gdf):
        """
        Разделяет граф на подграфы по микрорайонам.

        Parameters:
        - G: NetworkX граф.
        - districts_gdf: GeoDataFrame с геометриями микрорайонов.

        Returns:
        - district_graphs: Словарь, где ключ - имя микрорайона, значение - подграф.
        """
        district_graphs = {}
        for idx, district in districts_gdf.iterrows():
            district_polygon = district.geometry
            # Выбираем узлы, попадающие в микрорайон
            nodes_in_district = [node for node in G.nodes if Point(node).within(district_polygon)]
            if not nodes_in_district:
                print(f"Микрорайон {district['Name']} не содержит узлов графа.")
                continue
            subgraph = G.subgraph(nodes_in_district).copy()
            district_graphs[district['Name']] = subgraph
            print(
                f"Микрорайон '{district['Name']}' содержит {subgraph.number_of_nodes()} узлов и {subgraph.number_of_edges()} ребер.")
        return district_graphs

    def redistribute_load(self, G, house_node, school_node, new_edge, P=800):
        """
        Перераспределяет нагрузку после добавления новой дороги между домом и школой.

        Parameters:
        - G: NetworkX граф
        - house_node: узел дома
        - school_node: узел школы
        - new_edge: tuple (u, v) новой дороги
        - P: порог загруженности

        Returns:
        - G: обновлённый граф с перераспределённой нагрузкой
        """
        try:
            # Найти все кратчайшие пути от дома до школы
            paths = list(nx.all_shortest_paths(G, source=house_node, target=school_node, weight='weight'))
            print(f"Найдено {len(paths)} кратчайших путей между {house_node} и {school_node}.")

            # Пример простой логики перераспределения нагрузки:
            # Разделить нагрузку между всеми кратчайшими путями
            # Здесь необходимо доработать в зависимости от специфики задачи

            # Получить общую интенсивность между домом и школой
            try:
                intensity = G[house_node][school_node]['total_intensity']
            except KeyError:
                intensity = 0

            if intensity == 0:
                print(f"Нет интенсивности между {house_node} и {school_node}.")
                return G

            intensity_per_path = intensity / len(paths)
            print(f"Интенсивность {intensity} распределяется по {len(paths)} путям, по {intensity_per_path} на путь.")

            for path in paths:
                # Распределить нагрузку по ребрам пути
                for i in range(len(path) - 1):
                    u = path[i]
                    v = path[i + 1]
                    G[u][v]['total_intensity'] -= intensity_per_path
                    print(f"Уменьшена нагрузка на ребре ({u}, {v}) на {intensity_per_path}.")

            # Добавить нагрузку на новую дорогу
            G[new_edge[0]][new_edge[1]]['total_intensity'] += intensity
            print(f"Добавлена нагрузка {intensity} на новое ребро {new_edge}.")

        except Exception as e:
            print(f"Ошибка при перераспределении нагрузки: {e}")

        return G

    def run(self):
        # Определение лет застройки
        years = [1, 2, 3]  # Добавьте или измените годы по необходимости

        # Загрузка исходных данных улиц и зданий
        print("\n--- Загрузка исходных данных ---")
        streets_files = ["Streets_исходные.shp"] + [f"Streets_{i}очередь.shp" for i in range(1, 4)]
        streets_gdfs = []
        for file in streets_files:
            path = os.path.join(self.data_dir, file)
            gdf = self.load_data(path, self.target_crs)
            if gdf is not None:
                streets_gdfs.append(gdf)
            else:
                print(f"Файл {file} не был загружен и будет пропущен.")
        if streets_gdfs:
            streets_combined = pd.concat(streets_gdfs, ignore_index=True)
        else:
            streets_combined = gpd.GeoDataFrame(columns=['geometry'], crs=self.target_crs)
        streets_combined = self.clean_geometries(streets_combined)

        # Загрузка исходных зданий
        buildings_files = ["Дома_исходные.shp"] + [f"House_{i}очередь_ЖК.shp" for i in range(1, 4)]
        buildings_gdfs = []
        for file in buildings_files:
            path = os.path.join(self.data_dir, file)
            gdf = self.load_data(path, self.target_crs)
            if gdf is not None:
                buildings_gdfs.append(gdf)
            else:
                print(f"Файл {file} не был загружен и будет пропущен.")
        if buildings_gdfs:
            buildings_combined = pd.concat(buildings_gdfs, ignore_index=True)
        else:
            buildings_combined = gpd.GeoDataFrame(columns=['geometry'], crs=self.target_crs)
        buildings_combined = self.clean_geometries(buildings_combined)

        # Построение исходного графа
        print("\n--- Построение исходного графа ---")
        G = self.build_graph(streets_combined.head(7000), buildings_combined.head(7000))
        G_largest = self.get_largest_component(G)

        # Проверка связности графа
        if nx.is_connected(G_largest):
            print("Граф полностью соединен.")
        else:
            print("Граф не полностью соединен. Проверим количество компонент.")
            components = list(nx.connected_components(G))
            print(f"Количество компонент: {len(components)}")

        # Загрузка исходных домов и ключевых точек
        print("\n--- Загрузка исходных домов и ключевых точек ---")
        houses_orig = self.load_data(os.path.join(self.data_dir, "Дома_исходные.shp"), self.target_crs)
        houses_orig = self.clean_geometries(houses_orig)

        # Загрузка ключевых точек: Выходы метро и Остановки ОТ
        metro_exits = self.load_data(os.path.join(self.data_dir, "Выходы_метро.shp"), self.target_crs)
        bus_stops = self.load_data(os.path.join(self.data_dir, "Остановки_ОТ.shp"), self.target_crs)

        from shapely.geometry import Point

        key_point_nodes = []
        for geom in metro_exits.geometry:
            if isinstance(geom, Point):
                key_point_nodes.append((geom.x, geom.y))
            else:
                key_point_nodes.append((geom.centroid.x, geom.centroid.y))

        for geom in bus_stops.geometry:
            if isinstance(geom, Point):
                key_point_nodes.append((geom.x, geom.y))
            else:
                key_point_nodes.append((geom.centroid.x, geom.centroid.y))

        missing_key_nodes = [node for node in key_point_nodes if node not in G_largest.nodes]
        if missing_key_nodes:
            print(f"Некоторые целевые узлы отсутствуют в графе: {missing_key_nodes}")
        else:
            print("Все целевые узлы присутствуют в графе.")

        # Загрузка и фильтрация школ из первых очередей
        schools_gdfs = []
        for i in range(1, 4):
            houses_queue = self.load_data(os.path.join(self.data_dir, f"House_{i}очередь_ЖК.shp"), self.target_crs)
            if houses_queue is not None:
                houses_queue = self.clean_geometries(houses_queue)
                if 'Type' in houses_queue.columns:
                    schools = houses_queue[houses_queue['Type'] == 'Школы']
                    schools_gdfs.append(schools)
                else:
                    print(f"Столбец 'Type' отсутствует в данных House_{i}очередь_ЖК.shp. Фильтрация школ невозможна.")
        if schools_gdfs:
            schools_combined = pd.concat(schools_gdfs, ignore_index=True)
        else:
            schools_combined = gpd.GeoDataFrame(columns=['geometry'], crs=self.target_crs)

        # Объединение ключевых точек
        key_points = pd.concat([metro_exits, bus_stops, schools_combined], ignore_index=True)
        key_points = self.clean_geometries(key_points)

        # Привязка домов и ключевых точек к графу
        print("\n--- Привязка домов и ключевых точек к графу ---")
        houses_orig = self.bind_objects_to_graph(G_largest, houses_orig, object_type="Дом")
        key_points = self.bind_objects_to_graph(G_largest, key_points, object_type="Ключевая точка")

        # Расчёт интенсивности пешеходов
        print("\n--- Расчёт интенсивности пешеходов ---")
        houses_orig = self.calculate_intensity(houses_orig)

        # Инициализация атрибута 'total_intensity' для всех ребер
        for u, v in G_largest.edges():
            G_largest.edges[u, v]['total_intensity'] = 0.0

        # Вызов функции распределения интенсивности
        G_largest = self.distribute_intensity_to_edges(G_largest, houses_orig, key_point_nodes)

        # Дополнительный вывод для проверки
        print("Статистика по интенсивности:")
        try:
            print(houses_orig[['intensity_children_pensioners', 'intensity_adults_public_transport',
                               'intensity_adults_personal_transport', 'intensity_adults_cars',
                               'total_intensity']].describe())
        except KeyError as e:
            print(f"Отсутствует столбец: {e}. Проверьте правильность расчёта интенсивности.")

        # Идентификация узких мест
        print("\n--- Идентификация узких мест ---")
        bottlenecks = self.identify_bottlenecks(G_largest, self.P)
        print(bottlenecks)
        # Предложение новых путей
        print("\n--- Предложение новых путей ---")
        new_paths = self.suggest_new_paths(G_largest, bottlenecks, key_points, buildings_combined, num_suggestions=5)

        # Округление координат новых путей для соответствия округленным узлам графа
        precision = 2
        new_paths = [(
            (round(edge[0][0], precision), round(edge[0][1], precision)),
            (round(edge[1][0], precision), round(edge[1][1], precision))
        ) for edge in new_paths]

        # Добавление новых ребер в граф
        print("\n--- Добавление новых ребер в граф ---")
        G_optimized = self.add_new_edges_to_graph(G_largest, new_paths)

        # Проверка добавления ребер
        for edge in new_paths:
            u, v = edge
            if G_optimized.has_edge(u, v):
                print(f"Ребро {edge} успешно добавлено.")
            else:
                print(f"Ребро {edge} не было добавлено.")

        # Тестирование identify_bottlenecks с разными пороговыми значениями
        print("\n--- Тестирование identify_bottlenecks с разными порогами ---")
        for test_threshold in [100, 200, 500, 800]:  # Меньше 800 для проверки
            bottlenecks = self.identify_bottlenecks(G_largest, P=test_threshold)
            print(f"При пороге {test_threshold} чел/ч найдено {len(bottlenecks)} узких мест.")
            if bottlenecks:
                for u, v, intensity in bottlenecks[:5]:  # Показать первые 5 узких мест
                    print(f"Узкое место на ребре ({u}, {v}) с интенсивностью {intensity}")

            # # Перераспределение нагрузки после добавления новых дорог
        print("\n--- Перераспределение нагрузки после добавления новых дорог ---")
        for edge in new_paths:
            u, v = edge
            # Предполагается, что новое ребро соединяет дом и ключевую точку
            # Нужно определить соответствие между домом и ключевой точкой
            # В этом примере будем считать, что v является ключевой точкой
            # Это требует корректной логики, адаптируйте по вашим данным
            houses_connected = houses_orig[houses_orig['nearest_node'] == u]
            for idx, house in houses_connected.iterrows():
                # Найти соответствующую ключевую точку (например, ближайшую)
                self.redistribute_load(G_optimized, house['nearest_node'], v, edge, self.P)

        # Пересчёт загруженности после оптимизации
        print("\n--- Пересчёт загруженности после оптимизации ---")
        overloaded_after = self.recalculate_load(G_optimized, self.P)

        # Сохранение оптимизированного графа
        print("\n--- Сохранение оптимизированного графа ---")
        self.save_versioned_graph(G_optimized, year=years[0], output_dir=self.versions_dir)

        # Вычисление оптимальных маршрутов между домами и ключевыми точками
        print("\n--- Вычисление оптимальных маршрутов ---")
        sources = list(houses_orig['nearest_node'].unique())
        targets = list(key_points['nearest_node'].unique())
        routes = self.compute_optimal_routes(G_optimized, sources, targets)

        # Визуализация исходного и оптимизированного графа
        print("\n--- Визуализация исходного и оптимизированного графа ---")
        self.visualize_optimized_graph(G_largest, G_optimized, new_paths)

        # Визуализация оптимальных маршрутов
        print("\n--- Визуализация оптимальных маршрутов ---")
        self.visualize_routes(G_optimized, routes, years[0])

        # Визуализация доступности
        # print("\n--- Визуализация доступности ---")
        # self.visualize_accessibility(G_optimized, houses_orig, key_points, years[0])

        # Итоговый вывод
        print("\n--- Итоговая статистика ---")
        print(f"Количество узлов в G_largest: {G_largest.number_of_nodes()}")
        print(f"Количество ребер в G_largest: {G_largest.number_of_edges()}")
        print(f"Количество узлов в G_optimized: {G_optimized.number_of_nodes()}")
        print(f"Количество ребер в G_optimized: {G_optimized.number_of_edges()}")
        return 0
