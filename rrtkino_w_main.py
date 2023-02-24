from rrt_star import *
import numpy as np
from scipy.linalg import expm, inv
from scipy.integrate import quad
from numpy.linalg import matrix_power

class RRT_Star_Kyno(RRT_Star):
    def __init__(self, env = None, x_start= None, x_goal= None, step_len= None,
                 goal_sample_rate= None, search_radius= None, iter_max= None, r_RRT= None, r_goal= None, stop_at= None):

        self.name = 'RRTK*'
        self.state_dims = 4
        self.input_dims = 2
        self.A = np.zeros((self.state_dims,self.state_dims), dtype=np.double)
        self.A[0,2] = 1.0
        self.A[1,3] = 1.0
        self.B = np.zeros((self.state_dims, self.input_dims), np.double)
        self.B[2,0] = 1.0
        self.B[3,1] = 1.0
        self.R = np.eye(self.input_dims)
        self.c = np.zeros((self.input_dims,1))

    def init(self):
        cMin, theta = self.get_distance_and_angle(self.x_start, self.x_goal)
        C = self.RotationToWorldFrame(self.x_start, self.x_goal, cMin)
        xCenter = np.array([[(self.x_start.x + self.x_goal.x) / 2.0],
                            [(self.x_start.y + self.x_goal.y) / 2.0]])
        x_best = self.x_start

        return theta, cMin, xCenter, C, x_best

    def solution(self, t, t_s): #y(t)
        d = self.distance(t, t_s)
        solution_ = lambda x: (expm([[self.A, self.B @ inv(self.R) @ self.B.T],
                               [np.zeros((self.state_dims, self.state_dims)), -self.A]] *
                                (t - x)) @ np.vstack((self.c, np.zeros((self.state_dims, 1)))))

        sol = expm([[self.A, self.B @ inv(self.R) @ self.B.T],
                         [np.zeros((self.state_dims, self.state_dims)), -self.A.T]] *
                        (t - t_s)) @ np.vstack((self.x1, d)) + \
                        quad(solution_, t_s, t)
        
        return sol

    def distance(self, t, t_s):
        G_ = lambda x: expm(self.A * (t - x)) @ self.B @ inv(self.R) @ self.B.T @ expm(self.A.T * (t - x))
        G = quad(G_, t, t_s)

        x_bar_ = lambda x: expm(self.A * (t - x)) @ self.c
        x_bar = expm(self.A * t) @ self.x0 + quad(x_bar_, t, t_s)

        d = inv(G) * (self.x1 - x_bar)
        return d

    def states_eq(self, t, t_s):
        sol = self.solution(t, t_s)
        return sol[0:self.state_dims]
    
    def control(self, t, t_s):
        sol = self.solution(t, t_s)
        return inv(self.R) @ self.B.T @ sol[self.state_dims:2*self.state_dims,:]
    
    def sq_distance(self, t, t_s, dist_idxs):
        return matrix_power(self.x1[dist_idxs] - self.states_eq(t, t_s)[dist_idxs], 2).sum() - self.min_dist**2

    def cost_eq(self, t, t_s):
        cost_ = lambda x: 1 + (self.control(x, t_s) @ self.R @ self.control(x, t_s))
        return quad(cost_, 0, t_s)
    
    def tau_star(self, t, t_s):
        d = self.distance(t, t_s)
        return 1 - 2 * (self.A @ self.x1 + self.c).T * d - d.T * inv(self.B) @ self.R @ self.B.T * d

    def planning(self):
        theta, dist, x_center, C, x_best = self.init()
        c_best = np.inf
   
        x_best = self.x_start
        i = 0
        while i<self.iter_max and c_best >= self.stop_at:

            x_rand = self.SampleFromSpace()
            x_nearest = self.Nearest(x_rand)
            x_new = self.Steer(x_nearest, x_rand)

            if not self.env.is_collision(x_nearest, x_new):
                X_near = self.Near(self.V, x_new, self.r_RRT) # r_RRT
                c_min = self.Cost(x_nearest, x_new)

                # choose parent
                x_new, _ = self.choose_parent(X_near, x_new, c_min)

                self.V.append(x_new)

                # rewire
                self.rewire(X_near, x_new)

                x = self.V[self.V.index(x_new)]
                
                if x.equals(self.x_goal):
                    print('entrato')
                    x_best = x
                    c_best = self.Cost(x)

            # print("iter",i)
            if i % 20 == 0 or i == self.iter_max-1:
                # print("iter", i)
                plot(self, i)
            i+=1

        print("c_best", c_best)
        x_best, c_best = self.search_best()
        print("c_best", c_best)
        self.path = self.ExtractPath(x_best)
        # print("path", self.path)
        plot(self, 1)
        plt.plot([x for x, _ in self.path], [y for _, y in self.path], color=(1.0,0.0,0.0,1.0))
        plt.pause(0.001)
        plt.show()

    #initializes a new node in the direction of x_goal, distant at most step_len
    #from x_start
    def Steer(self, x_start, x_goal):
        dist, theta = self.get_distance_and_angle(x_start, x_goal)
        # print("dist", self.step_len, dist)
        dist = min(self.step_len, dist)

        node_new = Node((x_start.x + dist * math.cos(theta),
                         x_start.y + dist * math.sin(theta)))
        node_new.parent = x_start

        return node_new

    def Nearest(self, x_rand):
        return self.V[int(np.argmin([(n.x - x_rand.x) ** 2 + (n.y - x_rand.y) ** 2 for n in self.V]))]

    def Near(self, V, x_new, search_radius = 20):
        n = len(V) + 1
        r = min((search_radius * math.sqrt((math.log(n) / n))), self.step_len)
        #print("r2",r2)
        # if self.name == 'IRRT*':
        #     self.step_len = r
        r2 = r**2
        dist_table = [(n.x - x_new.x) ** 2 + (n.y - x_new.y) ** 2 for n in V]
        X_near = [v for v in V if dist_table[V.index(v)] <= r2 and not self.env.is_collision(v, x_new)]
        
        return X_near


    def inside_boundaries(self, x_rand):
        if x_rand is not None:
            return self.x_range[0] + self.delta <= x_rand[0] <= self.x_range[1] - self.delta and \
                    self.y_range[0] + self.delta <= x_rand[1] <= self.y_range[1] - self.delta
        else:
            return False

    def choose_parent(self, X_near, x_new, c_min):
        for x_near in X_near:
            c_new = self.Cost(x_near, x_new)
            if c_new < c_min and not self.env.is_collision(x_near, x_new):
                x_new.parent = x_near
                c_min = c_new
        return x_new, c_min

    def rewire(self, X_near, x_new):
        for x_near in X_near:
            c_near = self.Cost(x_near)
            c_new = self.Cost(x_new, x_near)
            if c_new < c_near:
                self.V[self.V.index(x_near)].parent = x_new
                # x_near.parent = x_new

    def search_best(self):
        distances = [(n.x - self.x_goal.x) ** 2 + (n.y - self.x_goal.y) ** 2 for n in self.V]
        # print("dist", distances)
        r2 = self.step_len**2
        indeces = [i for i in range(len(distances)) if distances[i] <= r2]
        
        if len(indeces)==0:
            return self.x_goal, np.inf 
        cost, idxs = zip(*[[self.Cost(self.V[idx], self.x_goal), idx] for idx in indeces])
        c_i = np.argmin(np.array(cost))
        best_index = idxs[c_i]
        x_best = self.V[best_index]
        c_best = cost[c_i]
        return x_best, c_best



    def SampleFromSpace(self):
        delta = self.delta

        if np.random.random() > self.goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return self.x_goal


    def ExtractPath(self, node):
        path = [[self.x_goal.x, self.x_goal.y]]

        while node.parent:
            # print(node.parent.x,node.parent.y)
            path.append([node.x, node.y])
            node = node.parent

        path.append([self.x_start.x, self.x_start.y])

        return path

    @staticmethod
    def RotationToWorldFrame(x_start, x_goal, L):
        # axes of the hyperellipsoid
        a1 = np.array([[(x_goal.x - x_start.x) / L],
                       [(x_goal.y - x_start.y) / L]])

        # first column of the identity matrix
        e1 = np.array([[1.0], [0.0]])

        M = a1 @ e1.T

        U, _, V_T = np.linalg.svd(M, full_matrices=True, compute_uv=True)
        C = U @ np.diag([1.0, np.linalg.det(U) * np.linalg.det(V_T.T)]) @ V_T

        return C


    @staticmethod
    def Line(x_start, x_goal):
        return math.hypot(x_goal.x - x_start.x, x_goal.y - x_start.y)


    def Cost(self, node, node2 = None):
        if node == self.x_start:
            return 0.0

        if node.parent is None:
            return np.inf

        if node2 is None:
            cost = 0.0
        else:
            cost = self.Line(node, node2)
        while node.parent:
            cost += self.Line(node, node.parent)
            node = node.parent

        return cost


    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)



def main():
    # rrtkino = RRT_Star_Kyno()
    # print(rrtkino.distance(0, 3))

    A = np.zeros((4,4), dtype=np.double)
    A[0,2] = 1.0
    A[1,3] = 1.0
    # res = np.zeros((4,4))
    # for i in range(A.shape[0]):
    #     for j in range(A.shape[1]):
    #         f = lambda x: A[i,j]*x
    #         res[i,j] = quad(f, 0.0, 3.0)[0]
            
    # print(res)
    def g(y):
        f = lambda x: y*x
        return quad(f, 0.0, 3.0)[0]
    gg = np.vectorize(g)
    print(gg(A))
if __name__ == '__main__':
    main()
