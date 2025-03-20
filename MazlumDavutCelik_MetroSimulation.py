from collections import defaultdict, deque
import heapq
from typing import Dict, List, Set, Tuple, Optional
from itertools import count
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


class Station:
    """
    Class representing a metro station.

    This class is used to model each station in the metro system. Every station contains
    a unique identifier (id), name, and information about the line it belongs to. Additionally,
    it stores the neighboring stations (other directly connected stations) and travel times to these
    stations. It can be considered as a node in a graph structure.

    Transitions between stations are modeled bidirectionally. If one can travel from one station
    to another, one can also travel from the latter to the former. This class forms the basic
    data structure for pathfinding algorithms on the metro network.

    Attributes:
        id (str): Unique identifier of the station (e.g., "K1", "M2")
        name (str): Name of the station (e.g., "Kızılay", "Ulus")
        line (str): Line that the station belongs to (e.g., "Red Line", "Blue Line")
        neighbors (List[Tuple['Station', int]]): Neighboring stations and travel times to them (in minutes)
    """

    def __init__(self, id: str, name: str, line: str):
        """
        Constructor for the Station class.

        Creates a new station with a unique identifier, name, and line information.
        The neighbors list is initialized as empty and will be populated using
        the add_neighbor method.

        Args:
            id (str): Unique identifier for the station (e.g., "K1", "M2")
            name (str): Name of the station (e.g., "Kizilay", "Ulus")
            line (str): Line that the station belongs to (e.g., "Kırmızı Hat", "Mavi Hat")
        """
        self.id = id
        self.name = name
        self.line = line
        self.neighbors: List[Tuple['Station', int]] = []  # (station, duration) tuples

    def add_neighbor(self, station: 'Station', duration: int):
        """
        Add a neighboring station to this station.

        This method establishes a one-way connection from this station to the specified
        neighboring station with a given travel duration. For a bidirectional connection,
        add_neighbor should be called for both stations.

        Args:
            station (Station): The neighboring station to add
            duration (int): Travel time between stations in minutes

        Example:
            # Creating a connection between two stations
            station1.add_neighbor(station2, 5)  # 5-minute connection from station1 to station2
        """
        self.neighbors.append((station, duration))


class MetroNetwork:
    """
    Class representing a metro transportation network.

    This class models the entire metro network as a graph, managing the relationships
    between stations and lines. It provides methods for adding stations and connections,
    as well as algorithms for finding optimal routes based on different criteria such as
    minimizing transfers or travel time.

    The network is implemented as a directed graph where stations are nodes and connections
    between stations are edges with weights representing travel times. The class also tracks
    which stations belong to which lines to facilitate transfer calculations.

    Algorithms implemented:
    - Breadth-First Search (BFS): Used in find_min_transfers method to find routes with
      minimum number of line transfers between stations.
    - A* Search: Used in find_fastest_route method to find the fastest route (lowest total
      travel time) between stations. Uses a simple heuristic function that prioritizes
      staying on the same line when possible.

    Both algorithms handle transfers between different metro lines and can determine optimal
    paths through the entire network.

    Attributes:
        stations (Dict[str, Station]): Dictionary mapping station IDs to Station objects
        lines (Dict[str, List[Station]]): Dictionary mapping line names to lists of stations on that line
    """

    def __init__(self):
        """
        Constructor for the MetroNetwork class.

        Initializes a new Metro Network with empty dictionaries for stations and lines.
        The stations dictionary will store Station objects indexed by their unique IDs,
        while the lines dictionary will store lists of stations grouped by line names.

        Example:
            # Create a new metro network
            metro = MetroNetwork()
        """

        self.stations: Dict[str, Station] = {}
        self.lines: Dict[str, List[Station]] = defaultdict(list)

    def add_station(self, id: str, name: str, line: str) -> None:
        """"
        Add a new station to the metro network.

        Creates a new Station object with the given parameters and adds it to both
        the stations dictionary (indexed by ID) and the appropriate line list in
        the lines dictionary. If a station with the same ID already exists, no
        new station will be created.

        Args:
            id (str): Unique identifier for the station (e.g., "K1", "M2")
            name (str): Name of the station (e.g., "Kizilay", "Ulus")
            line (str): Line that the station belongs to (e.g., "Kırmızı Hat", "Mavi Hat")

        Example:
            # Add Kızılay station to the line named "Kırmızı Hat"
            metro.add_station("K1", "Kizilay", "Red Line")
        """
        if id not in self.stations:
            station = Station(id, name, line)
            self.stations[id] = station
            self.lines[line].append(station)

    def add_connection(self, station1_id: str, station2_id: str, duration: int) -> None:
        """
        Create a bidirectional connection between two stations in the network.

        Establishes a two-way connection between the stations identified by station1_id
        and station2_id, with the specified travel duration. This method adds each station
        to the other's neighbors list, creating a bidirectional edge in the graph.

        Args:
            station1_id (str): ID of the first station
            station2_id (str): ID of the second station
            duration (int): Travel time between stations in minutes

        Note:
            Both stations must already exist in the network, otherwise a KeyError may be raised.

        Example:
            # Connect station1 (station1 id is "K1") and station2 (station2 id is "K2") stations with a 4-minute travel time
            metro.add_connection("K1", "K2", 4)
        """
        station1 = self.stations[station1_id]
        station2 = self.stations[station2_id]
        station1.add_neighbor(station2, duration)
        station2.add_neighbor(station1, duration)

    def find_min_transfers(self, start_id: str, destination_id: str) -> Optional[Tuple[Station]]:
        """
        Find a route with the minimum number of line transfers between two stations.

        This method uses a Breadth-First Search (BFS) algorithm to find the path between
        start_id and destination_id that requires the fewest line changes. Line transfers
        are counted whenever two consecutive stations in the path belong to different lines.

        Args:
            start_id (str): ID of the starting station
            destination_id (str): ID of the destination station

        Returns:
            Optional[Tuple[List[Station], int]]: If a path exists, returns a tuple containing
                the list of stations in the path and the number of transfers required.
                If no path exists, returns None.

        Example:
            # Find minimum transfer route from station1 (station1 id is "K1") to station2 (station2 id is "T4")
            route, transfers = metro.find_min_transfers("K1", "T4")
            if route:
                print(f"Route requires {transfers} transfers")
            """
        if start_id not in self.stations or destination_id not in self.stations:
            return None

        start = self.stations[start_id]
        destination = self.stations[destination_id]

        # A dictionary to keep track of the stations visited and
        # the minimum number of transfers required to reach that station.
        visited = {start_id: 0}  # (station_id: transfers_count)


        queue = deque([(start, 0, [start])]) # Queue elements: (station, transfers_count, path)

        while queue:
            current, transfers_count, path = queue.popleft()

            if current.id == destination_id: # If we reached the destination, return the path
                return path, transfers_count

            for neighbor, _ in current.neighbors: # For every neighbor
                new_transfers_count = transfers_count
                if current.line != neighbor.line: # Check if there is a line change
                    new_transfers_count += 1
                # If this neighbor has not been reached before or can be reached with fewer transfers
                if neighbor.id not in visited or new_transfers_count < visited[neighbor.id]:
                    visited[neighbor.id] = new_transfers_count
                    queue.append((neighbor, new_transfers_count, path + [neighbor]))
        return None

    def find_fastest_route(self, start_id: str, destination_id: str) -> Optional[Tuple[List[Station], int]]:
        """
            Find the fastest route (minimum travel time) between two stations using A* algorithm.

            This method implements the A* search algorithm to find the optimal path with the lowest
            total travel time between the start and destination stations. The algorithm considers both
            direct travel times between stations and potential transfer times when changing lines.

            Args:
                start_id (str): ID of the starting station
                destination_id (str): ID of the destination station

            Returns:
                Optional[Tuple[List[Station], int]]: If a path exists, returns a tuple containing
                    the list of stations in the path and the total travel time in minutes.
                    If no path exists, returns None.

            Example:
                # Find fastest route from station1 (station id is K1) to station2 (station2 id is K4)
                route, total_time = metro.find_fastest_route("K1", "K4")
                if route:
                    print(f"Fastest route takes {total_time} minutes")
        """
        if start_id not in self.stations or destination_id not in self.stations:
            return None

        start = self.stations[start_id]
        destination = self.stations[destination_id]

        # First, let's calculate the shortest paths between all pairs using Floyd-Warshall algorithm for
        # the heuristic that calculates the minimum time from each station to the destination
        min_durations = self._calculate_minimum_durations()

        # Counter to be used to compare equal f values for the priority queue
        c = count()

        # Open list (priority queue) - (f value, counter, station, path, total duration)
        # f value = g value (time elapsed so far) + h value (estimated time to destination)
        open_list = [(min_durations.get((start.id, destination.id), float('inf')), next(c), start, [start], 0)]
        heapq.heapify(open_list)

        # Dictionary to store the best g value (time from start to station) for each station
        g_values = {station.id: float('inf') for station in self.stations.values()}
        g_values[start.id] = 0

        # To keep of visited stations
        visited = set()

        while open_list:
            # Get the station with the lowest f value
            _, _, current, path, total_duration = heapq.heappop(open_list)

            # If we reached the destination station, return the route and total time
            if current.id == destination.id:
                    return (path, total_duration)

            # If the station has already been visited and a better way has been found, skip it
            if current.id in visited and g_values[current.id] <= total_duration:
                continue

            # Mark station as visited
            visited.add(current.id)

            #Check all neighbors of current station
            for neighbor, duration in current.neighbors:
                # New g value (total duration from start to neighbor)
                new_duration = total_duration + duration

                # If the neighbor has already been reached by a shorter route, skip that route
                if neighbor.id in visited and g_values[neighbor.id] <= new_duration:
                    continue

                # If found a better way, update g value
                if new_duration < g_values[neighbor.id]:
                    g_values[neighbor.id] = new_duration

                    # Estimated total duration to neighbor (f value)
                    f_values = new_duration + min_durations.get((neighbor.id, destination.id), float('inf'))

                    # Add neighbor to open list
                    heapq.heappush(open_list, (f_values, next(c), neighbor, path + [neighbor], new_duration))
        # If no route is found, return None
        return None

    def _calculate_minimum_durations(self) -> Dict[Tuple[str, str], int]:
        """
        Calculates the minimum durations between all station pairs using the Floyd-Warshall algorithm.
        This is used as the optimal heuristic function for the A* algorithm.

        Returns:
            Dict[Tuple[str, str], int]: Dictionary containing the minimum duration for each station pair
                                    Key is the pair (station1_id, station2_id), value is the duration
        """
        # Infinite initial value for all station pairs
        min_durations = {}
        for i in self.stations.values():
            for j in self.stations.values():
                if i.id == j.id:
                    min_durations[(i.id, j.id)] = 0
                else:
                    min_durations[(i.id, j.id)] = float('inf')

        # Add direct links
        for station in self.stations.values():
            for neighbor, duration in station.neighbors:
                min_durations[(station.id, neighbor.id)] = min(min_durations[(station.id, neighbor.id)], duration)

        # Floyd-Warshall algorithm
        for k in self.stations.values():
            for i in self.stations.values():
                for j in self.stations.values():
                    if min_durations[(i.id, k.id)] + min_durations[(k.id, j.id)] < min_durations[(i.id, j.id)]:
                        min_durations[(i.id, j.id)] = min_durations[(i.id, k.id)] + min_durations[(k.id, j.id)]

        return min_durations

# Example Usage
if __name__ == "__main__":
    metro = MetroNetwork()
    def find_transfer_stations(route: List[Station]) -> List[Station]:
        """
        Find the transfer stations in a given route.

        This function identifies stations where passengers need to change from one line to another
        in a given route. A transfer stations is detected when two consecutive stations in the route
        belong to different lines.

        Args:
            route (List[Station]): A list of stations representing the complete route

        Returns:
            List[Station]: A list of stations where transfers between lines occur

        Example:
            # Find transfer stations in a route from Kızılay to Keçiören
            route, _ = metro.find_min_transfers("K1", "T4")
            transfer_stations = find_transfer_stations(route)
            print(f"You need to transfer at: {', '.join(station.name for station in transfer_stations)}")
        """
        transfer_stations = []
        for i in range(1, len(route)):
            if route[i - 1].line != route[i].line:
                transfer_stations.append(route[i - 1])  # Aktarma yapılan istasyon
        return transfer_stations

    # Adding stations
    # Kırmızı Hat
    metro.add_station("K1", "Kızılay", "Kırmızı Hat")
    metro.add_station("K2", "Ulus", "Kırmızı Hat")
    metro.add_station("K3", "Demetevler", "Kırmızı Hat")
    metro.add_station("K4", "OSB", "Kırmızı Hat")

    # Mavi Hat
    metro.add_station("M1", "AŞTİ", "Mavi Hat")
    metro.add_station("M2", "Kızılay", "Mavi Hat")  # Aktarma noktası
    metro.add_station("M3", "Sıhhiye", "Mavi Hat")
    metro.add_station("M4", "Gar", "Mavi Hat")

    # Turuncu Hat
    metro.add_station("T1", "Batıkent", "Turuncu Hat")
    metro.add_station("T2", "Demetevler", "Turuncu Hat")  # Aktarma noktası
    metro.add_station("T3", "Gar", "Turuncu Hat")  # Aktarma noktası
    metro.add_station("T4", "Keçiören", "Turuncu Hat")

    # Adding connection
    # "Kırmızı Hat" connections
    metro.add_connection("K1", "K2", 4)  # Kızılay -> Ulus
    metro.add_connection("K2", "K3", 6)  # Ulus -> Demetevler
    metro.add_connection("K3", "K4", 8)  # Demetevler -> OSB

    # "Mavi Hat" connections
    metro.add_connection("M1", "M2", 5)  # AŞTİ -> Kızılay
    metro.add_connection("M2", "M3", 3)  # Kızılay -> Sıhhiye
    metro.add_connection("M3", "M4", 4)  # Sıhhiye -> Gar

    # "Turuncu Hat" connections
    metro.add_connection("T1", "T2", 7)  # Batıkent -> Demetevler
    metro.add_connection("T2", "T3", 9)  # Demetevler -> Gar
    metro.add_connection("T3", "T4", 5)  # Gar -> Keçiören

    # Line transfer connections (same station different lines)
    metro.add_connection("K1", "M2", 2)  # Kızılay aktarma
    metro.add_connection("K3", "T2", 3)  # Demetevler aktarma
    metro.add_connection("M4", "T3", 2)  # Gar aktarma

    # Test Scenarios
    print("\n===========================================Test Scenarios ===========================================")
    print("\n#####################################################################################################")

    # Scenario 1: From AŞTİ to OSB
    print("\n1. From AŞTİ to OSB:")
    route = metro.find_min_transfers("M1", "K4")
    if route:
        route, transfers_count = route
        transfer_stations = find_transfer_stations(route)
        print(f"🚊Route with minimum transfers ({transfers_count} stop/s):\n",
              " -> ".join(f"{i.name}({i.line})" for i in route))
        print(f"🚦Transfer stations:\n {', '.join(transfer_station.name for transfer_station in transfer_stations)}")

    sonuc = metro.find_fastest_route("M1", "K4")
    if sonuc:
        route, sure = sonuc
        print(f"\n🚀The fastest route ({sure} minutes):\n", " -> ".join(f"{i.name}({i.line})" for i in route))
        print("\n#####################################################################################################")


    # Scenario 2: From Batıkent to Keçiören
    print("\n2: From Batıkent to Keçiören:")
    route = metro.find_min_transfers("T1", "T4")
    if route:
        route, transfers_count = route
        transfer_stations = find_transfer_stations(route)
        print(f"🚊Route with minimum transfers ({transfers_count} stop/s):\n",
              " -> ".join(f"{i.name}({i.line})" for i in route))
        print(f"🚦Transfer stations:\n {', '.join(transfer_station.name for transfer_station in transfer_stations)}")

    sonuc = metro.find_fastest_route("T1", "T4")
    if sonuc:
        route, sure = sonuc
        print(f"\n🚀The fastest route ({sure} minutes):\n", " -> ".join(f"{i.name}({i.line})" for i in route))
        print("\n#####################################################################################################")

    # Scenario 3: From Keçiören to AŞTİ
    print("\n3. From Keçiören to AŞTİ:")
    route = metro.find_min_transfers("T4", "M1")
    if route:
        route, transfers_count = route
        transfer_stations = find_transfer_stations(route)
        print(f"🚊Route with minimum transfers ({transfers_count} stop/s):\n",
              " -> ".join(f"{i.name}({i.line})" for i in route))
        print(f"🚦Transfer stations:\n {', '.join(transfer_station.name for transfer_station in transfer_stations)}")

    sonuc = metro.find_fastest_route("T4", "M1")
    if sonuc:
        route, sure = sonuc
        print(f"\n🚀The fastest route ({sure} minutes):\n", " -> ".join(f"{i.name}({i.line})" for i in route))
        print("\n#####################################################################################################")

    # Scenario 4: From OSB to AŞTİ
    print("\n4. From OSB to AŞTİ:")
    route = metro.find_min_transfers("K4", "M1")
    if route:
        route, transfers_count = route
        transfer_stations = find_transfer_stations(route)
        print(f"🚊Route with minimum transfers ({transfers_count} stop/s):\n",
              " -> ".join(f"{i.name}({i.line})" for i in route))
        print(f"🚦Transfer stations:\n {', '.join(transfer_station.name for transfer_station in transfer_stations)}")

    sonuc = metro.find_fastest_route("K4", "M1")
    if sonuc:
        route, sure = sonuc
        print(f"\n🚀The fastest route ({sure} minutes):\n", " -> ".join(f"{i.name}({i.line})" for i in route))
        print("\n#####################################################################################################")

    # Scenario 5: From AŞTİ to Keçiören
    print("\n5. From AŞTİ to Keçiören :")
    route = metro.find_min_transfers("M1", "T4")
    if route:
        route, transfers_count = route
        transfer_stations = find_transfer_stations(route)
        print(f"🚊Route with minimum transfers ({transfers_count} stop/s):\n",
              " -> ".join(f"{i.name}({i.line})" for i in route))
        print(f"🚦Transfer stations:\n {', '.join(transfer_station.name for transfer_station in transfer_stations)}")

    sonuc = metro.find_fastest_route("M1", "T4")
    if sonuc:
        route, sure = sonuc
        print(f"\n🚀The fastest route ({sure} minutes):\n", " -> ".join(f"{i.name}({i.line})" for i in route))
        print("\n#####################################################################################################")

    # Scenario 6: From AŞTİ to Batıkent
    print("\n6. From AŞTİ to Batıkent :")
    route = metro.find_min_transfers("M1", "T1")
    if route:
        route, transfers_count = route
        transfer_stations = find_transfer_stations(route)
        print(f"🚊Route with minimum transfers ({transfers_count} stop/s):\n",
              " -> ".join(f"{i.name}({i.line})" for i in route))
        print(f"🚦Transfer stations:\n {', '.join(transfer_station.name for transfer_station in transfer_stations)}")

    sonuc = metro.find_fastest_route("M1", "T1")
    if sonuc:
        route, sure = sonuc
        print(f"\n🚀The fastest route ({sure} minutes):\n", " -> ".join(f"{i.name}({i.line})" for i in route))
        print("\n#####################################################################################################")

    # Scenario 7: From Keçiören to OSB
    print("\n7. From Keçiören to OSB :")
    route = metro.find_min_transfers("T4", "K4")
    if route:
        route, transfers_count = route
        transfer_stations = find_transfer_stations(route)
        print(f"🚊Route with minimum transfers ({transfers_count} stop/s):\n",
              " -> ".join(f"{i.name}({i.line})" for i in route))
        print(f"🚦Transfer stations:\n {', '.join(transfer_station.name for transfer_station in transfer_stations)}")

    sonuc = metro.find_fastest_route("T4", "K4")
    if sonuc:
        route, sure = sonuc
        print(f"\n🚀The fastest route ({sure} minutes):\n", " -> ".join(f"{i.name}({i.line})" for i in route))
        print("\n#####################################################################################################")

    # Scenario 8: From Keçiören to Ulus
    print("\n8. From Keçiören to Ulus :")
    route = metro.find_min_transfers("T4", "K2")
    if route:
        route, transfers_count = route
        transfer_stations = find_transfer_stations(route)
        print(f"🚊Route with minimum transfers ({transfers_count} stop/s):\n",
              " -> ".join(f"{i.name}({i.line})" for i in route))
        print(f"🚦Transfer stations:\n {', '.join(transfer_station.name for transfer_station in transfer_stations)}")

    sonuc = metro.find_fastest_route("T4", "K2")
    if sonuc:
        route, sure = sonuc
        print(f"\n🚀The fastest route ({sure} minutes):\n", " -> ".join(f"{i.name}({i.line})" for i in route))
        print("\n#####################################################################################################")

    # Scenario 9: From Batıkent to Sıhhiye
    print("\n9. From Batıkent to Sıhhiye :")
    route = metro.find_min_transfers("T1", "M3")
    if route:
        route, transfers_count = route
        transfer_stations = find_transfer_stations(route)
        print(f"🚊Route with minimum transfers ({transfers_count} stop/s):\n",
              " -> ".join(f"{i.name}({i.line})" for i in route))
        print(f"🚦Transfer stations:\n {', '.join(transfer_station.name for transfer_station in transfer_stations)}")

    sonuc = metro.find_fastest_route("T1", "M3")
    if sonuc:
        route, sure = sonuc
        print(f"\n🚀The fastest route ({sure} minutes):\n", " -> ".join(f"{i.name}({i.line})" for i in route))
        print("\n#####################################################################################################")

    # Scenario 10: From OSB to Sıhhiye
    print("\n10. From OSB to Sıhhiye :")
    route = metro.find_min_transfers("K4", "M3")
    if route:
        route, transfers_count = route
        transfer_stations = find_transfer_stations(route)
        print(f"🚊Route with minimum transfers ({transfers_count} stop/s):\n",
              " -> ".join(f"{i.name}({i.line})" for i in route))
        print(f"🚦Transfer stations:\n {', '.join(transfer_station.name for transfer_station in transfer_stations)}")

    sonuc = metro.find_fastest_route("K4", "M3")
    if sonuc:
        route, sure = sonuc
        print(f"\n🚀The fastest route ({sure} minutes):\n", " -> ".join(f"{i.name}({i.line})" for i in route))
        print("\n#####################################################################################################")
