import java.util.*;

public class DijkstraSolver {
    static class Edge {
        int to, w;

        Edge(int to, int w) {
            this.to = to;
            this.w = w;
        }
    }

    static class State {
        int vertex, dist;

        State(int vertex, int dist) {
            this.vertex = vertex;
            this.dist = dist;
        }
    }

    public static DijkstraResult dijkstra(int n, List<int[]> edges, int start) {
        if (n <= 0)
            throw new IllegalArgumentException("Number of vertices must be positive");
        Objects.requireNonNull(edges, "Edges list must not be null");
        if (start < 0 || start >= n)
            throw new IllegalArgumentException("Invalid start vertex");

        List<List<Edge>> g = new ArrayList<>();
        for (int i = 0; i < n; i++)
            g.add(new ArrayList<>());
        for (int[] e : edges) {
            if (e.length != 3)
                throw new IllegalArgumentException("Edge must contain u,v,w");
            int u = e[0], v = e[1], w = e[2];
            g.get(u).add(new Edge(v, w));
            g.get(v).add(new Edge(u, w));
        }

        int[] dist = new int[n];
        int[] parent = new int[n];
        boolean[] vis = new boolean[n];
        Arrays.fill(dist, Integer.MAX_VALUE);
        Arrays.fill(parent, -1);
        dist[start] = 0;

        PriorityQueue<State> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a.dist));
        pq.offer(new State(start, 0));

        while (!pq.isEmpty()) {
            State cur = pq.poll();
            int u = cur.vertex;
            if (vis[u])
                continue;
            vis[u] = true;

            for (Edge e : g.get(u)) {
                int v = e.to;
                int newDist = dist[u] + e.w;
                if (newDist < dist[v]) {
                    dist[v] = newDist;
                    parent[v] = u;
                    pq.offer(new State(v, newDist));
                }
            }
        }

        return new DijkstraResult(dist, parent);
    }

    public static class DijkstraResult {
        public final int[] dist;
        public final int[] parent;

        DijkstraResult(int[] dist, int[] parent) {
            this.dist = dist;
            this.parent = parent;
        }

        public List<Integer> getPath(int target) {
            List<Integer> path = new ArrayList<>();
            for (int v = target; v != -1; v = parent[v])
                path.add(v);
            Collections.reverse(path);
            return path;
        }
    }

    public static void main(String[] args) {
        int n = 5;
        List<int[]> edges = new ArrayList<>();
        edges.add(new int[] { 0, 1, 2 });
        edges.add(new int[] { 0, 2, 4 });
        edges.add(new int[] { 1, 2, 1 });
        edges.add(new int[] { 1, 3, 7 });
        edges.add(new int[] { 2, 4, 3 });
        edges.add(new int[] { 3, 4, 2 });

        DijkstraResult res = dijkstra(n, edges, 0);
        System.out.println("Shortest distances from node 0:");
        for (int i = 0; i < n; i++) {
            System.out.printf("0 → %d = %d, path: %s%n",
                    i, res.dist[i], res.getPath(i));
        }
    }
}
