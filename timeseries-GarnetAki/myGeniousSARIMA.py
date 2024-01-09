import numpy as np
import numpy.linalg as la

class myModelSARIMA():

    def differentiation(data, d, D, s):
        saved_values = []
        data_tmp = data
        
        for i in range(D):
            for j in range(s):
                saved_values.append(data_tmp[j])
            new_data_tmp = []
            for j in range(len(data_tmp)):
                if j >= s:
                    new_data_tmp.append(data_tmp[j] - data_tmp[j-s])
            data_tmp = new_data_tmp

        for i in range(d):
            saved_values.append(data_tmp[0])
            new_data_tmp = []
            for j in range(len(data_tmp)):
                if j!=0:
                    new_data_tmp.append(data_tmp[j] - data_tmp[j-1])
            data_tmp = new_data_tmp

        return data_tmp, saved_values
        
    def integration(data, saved_values, d, D, s):
        data_tmp = data

        for i in range(d):
            new_data_tmp = []
            new_data_tmp.append(saved_values[-1])
            saved_values = saved_values[:-1]
            for j in range(len(data_tmp)):
                new_data_tmp.append(data_tmp[j] + new_data_tmp[j])
            data_tmp = new_data_tmp

        for i in range(D):
            new_data_tmp = []
            for j in range(s):
                new_data_tmp.append(saved_values[-s+j])
            saved_values = saved_values[:-s]
            for j in range(len(data_tmp)):
                new_data_tmp.append(data_tmp[j] + new_data_tmp[j])
            data_tmp = new_data_tmp

        return data_tmp

    def create_matrix_x(data, parameter, seas_parameter, s):
        matrix_template = []
        for i in range(1, len(data)+1, 1):
            row = [1]
            if parameter-i+1 > 0:
                row.extend(np.zeros(parameter-i+1))
                for j in range(1, i, 1):
                    row.append(data[j-1])
            else:
                for j in range(i - parameter, i, 1):
                    row.append(data[j-1])
            P_tmp = seas_parameter
            while P_tmp > 0:
                if P_tmp*s-i+1 > 0:
                    row.append(0)
                    P_tmp -= 1
                else:
                    for j in range(i - P_tmp*s, i, s):
                        row.append(data[j-1])
                    P_tmp = 0
            matrix_template.append(row)
        return np.array(matrix_template)

    def mgSARIMA(self, data, order, seasonal_order):
        p, d, q = order[0], order[1], order[2]
        P, D, Q, s = seasonal_order[0], seasonal_order[1], seasonal_order[2], seasonal_order[3]
        differentiated_data, saved_values = self.differentiation(data, d, D, s)
        matrix_p = self.create_matrix_x(differentiated_data, p, P, s)
        matrix_weight_p = (la.inv(matrix_p.T @ matrix_p)) @ matrix_p.T @ differentiated_data
        matrix_calculated_p = matrix_p @ matrix_weight_p
        matrix_err_p = differentiated_data - matrix_calculated_p
        matrix_q = self.create_matrix_x(matrix_err_p, q, Q, s)
        matrix_weight_q = (la.inv(matrix_q.T @ matrix_q)) @ matrix_q.T @ matrix_err_p
        matrix_calculated = matrix_p @ matrix_weight_p - matrix_q @ matrix_weight_q

        integrated_data = self.integration(matrix_calculated, saved_values, d, D, s)
        self.model_data = integrated_data
        self.weights_p = matrix_weight_p
        self.weights_q = matrix_weight_q
        self.model_errors = matrix_err_p
        self.params = [p, d, q, P, D, Q, s]

    def add_part_row(data, parameter, seas_parameter, s):
        row = [1]
        l = len(data)
        if parameter-l > 0:
            row.extend(np.zeros(parameter-l))
            for j in range(0, l, 1):
                row.append(data[j])
        else:
            for j in range(l - parameter, l, 1):
                row.append(data[j])
        P_tmp = seas_parameter
        while P_tmp > 0:
            if P_tmp*s-l > 0:
                row.append(0)
                P_tmp -= 1
            else:
                for j in range(l - P_tmp*s, l, s):
                    row.append(data[j])
                P_tmp = 0
        return np.array(row)

    def predict(self, n):
        model_data_with_predicted = self.model_data
        model_errors_with_predicted = self.model_errors
        predicted_values = []
        p, P, q, Q, s = self.params[0], self.params[3], self.params[2], self.params[5], self.params[6]
        for i in range(n):
            row_value = self.add_part_row(model_data_with_predicted, p, P, s)
            row_error = self.add_part_row(model_errors_with_predicted, q, Q, s)
            new_error = row_error @ self.weights_q
            new_value = row_value @ self.weights_p - new_error
            model_data_with_predicted = np.append(model_data_with_predicted, new_value)
            model_errors_with_predicted = np.append(model_errors_with_predicted, new_error)
            predicted_values.append(new_value)
        return predicted_values