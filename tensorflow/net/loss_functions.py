import tensorflow as tf
import keras.backend as K
from keras.losses import binary_crossentropy
#from sklearn.metrics import f1_score, precision_score, recall_score

beta = 0.25
alpha = 0.25
gamma = 2
epsilon = 1e-5
smooth = 1

def _to_tensor(x, dtype):
        """Convert the input `x` to a tensor of type `dtype`.
        # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.
        # Returns
        A tensor.
        """
        x = tf.convert_to_tensor(x)
        if x.dtype != dtype:
            x = tf.cast(x, dtype)
        return x



class Semantic_loss_functions(object):
    def __init__(self):
        print ("semantic loss functions initialized")

    def dice_coef(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + K.epsilon()) / (
                    K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

    def sensitivityo(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + K.epsilon())

    def specificity0(self, y_true, y_pred):
        true_negatives = K.sum(
            K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
        return true_negatives / (possible_negatives + K.epsilon())

    def convert_to_logits(self, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(),
                                  1 - tf.keras.backend.epsilon())
        return tf.math.log(y_pred / (1 - y_pred))

    def weighted_cross_entropyloss(self, y_true, y_pred):
        y_pred = self.convert_to_logits(y_pred)
        pos_weight = beta / (1 - beta)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred,
                                                        targets=y_true,
                                                        pos_weight=pos_weight)
        return tf.reduce_mean(loss)

    def focal_loss_with_logits(self, logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(
            -logits)) * (weight_a + weight_b) + logits * weight_b

    def focal_loss(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(),
                                  1 - tf.keras.backend.epsilon())
        logits = tf.math.log(y_pred / (1 - y_pred))

        loss = self.focal_loss_with_logits(logits=logits, targets=y_true,
                                      alpha=alpha, gamma=gamma, y_pred=y_pred)

        return tf.reduce_mean(loss)

    def depth_softmax(self, matrix):
        sigmoid = lambda x: 1 / (1 + K.exp(-x))
        sigmoided_matrix = sigmoid(matrix)
        softmax_matrix = sigmoided_matrix / K.sum(sigmoided_matrix, axis=0)
        return softmax_matrix

    def generalized_dice_coefficient(self, y_true, y_pred):
        smooth = 1
        y_true = K.clip(y_true, 0, 1)
        y_pred = K.clip(y_pred, 0, 1)
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (
                    K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return score
    
    '''def generalized_dice_coefficient(self, y_true, y_pred):
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (
                    K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return score'''


    def dice_loss(self, y_true, y_pred):
        loss = 1 - self.generalized_dice_coefficient(y_true, y_pred)
        return loss
    
    def dice(self, y_true, y_pred):
        loss = self.generalized_dice_coefficient(y_true, y_pred)
        return loss

    def bce_dice_loss(self, y_true, y_pred):
        loss = binary_crossentropy(y_true, y_pred) + \
               self.dice_loss(y_true, y_pred)
        return loss / 2.0

    def confusion(self, y_true, y_pred):
        smooth = 1
        y_pred_pos = K.clip(y_pred, 0, 1)
        y_pred_neg = 1 - y_pred_pos
        y_pos = K.clip(y_true, 0, 1)
        y_neg = 1 - y_pos
        tp = K.sum(y_pos * y_pred_pos)
        fp = K.sum(y_neg * y_pred_pos)
        fn = K.sum(y_pos * y_pred_neg)
        prec = (tp + smooth) / (tp + fp + smooth)
        recall = (tp + smooth) / (tp + fn + smooth)
        return prec, recall

    def true_positive(self, y_true, y_pred):
        smooth = 1
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        y_pos = K.round(K.clip(y_true, 0, 1))
        tp = (K.sum(y_pos * y_pred_pos) + smooth) / (K.sum(y_pos) + smooth)
        return tp

    def true_negative(self, y_true, y_pred):
        smooth = 1
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        y_pred_neg = 1 - y_pred_pos
        y_pos = K.round(K.clip(y_true, 0, 1))
        y_neg = 1 - y_pos
        tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth)
        return tn
    
    def cross_entropy_balanced(self,y_true, y_pred):
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits, 
    # Keras expects probabilities.
    # transform y_pred back to logits
        _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
        y_pred   = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
        y_pred   = tf.log(y_pred/ (1 - y_pred))

        y_true = tf.cast(y_true, tf.float32)

        count_neg = tf.reduce_sum(1. - y_true)
        count_pos = tf.reduce_sum(y_true)

        beta = count_neg / (count_neg + count_pos)

        pos_weight = beta / (1 - beta)

        cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

        cost = tf.reduce_mean(cost * (1 - beta))

        return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)

    def tversky_index(self, y_true, y_pred):
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        true_pos = K.sum(y_true_pos * y_pred_pos)
        false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
        alpha = 0.7
        return (true_pos + smooth) / (true_pos + alpha * false_neg + (
                    1 - alpha) * false_pos + smooth)

    def tversky_loss(self, y_true, y_pred):
        return 1 - self.tversky_index(y_true, y_pred)

    def focal_tversky(self, y_true, y_pred):
        pt_1 = self.tversky_index(y_true, y_pred)
        gamma = 0.75
        return K.pow((1 - pt_1), gamma)

    def log_cosh_dice_loss(self, y_true, y_pred):
        x = self.dice_loss(y_true, y_pred)
        return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)
    
    

    '''def precision(self,y_true, y_pred):
            """Precision metric.

            Only computes a batch-wise average of precision.

            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

    def recall(self,y_true, y_pred):
            """Recall metric.

            Only computes a batch-wise average of recall.

            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall'''
    

    '''def f1_socre(self,y_true, y_pred):
        def recall(y_true, y_pred):
            """Recall metric.

            Only computes a batch-wise average of recall.

            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall
        def precision(y_true, y_pred):
            """Precision metric.

            Only computes a batch-wise average of precision.

            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision
        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))'''

    '''def cal_base(self,y_true, y_pred):
        y_pred_positive = K.round(K.clip(y_pred, 0, 1))
        y_pred_negative = 1 - y_pred_positive

        y_positive = K.round(K.clip(y_true, 0, 1))
        y_negative = 1 - y_positive

        TP = K.sum(y_positive * y_pred_positive)
        TN = K.sum(y_negative * y_pred_negative)

        FP = K.sum(y_negative * y_pred_positive)
        FN = K.sum(y_positive * y_pred_negative)
        return TP, TN, FP, FN


    def acc(self,y_true, y_pred):
        y_pred_positive = K.round(K.clip(y_pred, 0, 1))
        y_pred_negative = 1 - y_pred_positive

        y_positive = K.round(K.clip(y_true, 0, 1))
        y_negative = 1 - y_positive

        TP = K.sum(y_positive * y_pred_positive)
        TN = K.sum(y_negative * y_pred_negative)

        FP = K.sum(y_negative * y_pred_positive)
        FN = K.sum(y_positive * y_pred_negative)
        ACC = (TP + TN) / (TP + FP + FN + TN + K.epsilon())
        return ACC'''


    def sensitivity(self,y_true, y_pred):
        """ recall """
        y_pred_positive = K.round(K.clip(y_pred, 0, 1))
        y_pred_negative = 1 - y_pred_positive

        y_positive = K.round(K.clip(y_true, 0, 1))
        #y_negative = 1 - y_positive

        TP = K.sum(y_positive * y_pred_positive)
        FN = K.sum(y_positive * y_pred_negative)
        SE = TP/(TP + FN + K.epsilon())
        return SE


    def precision(self,y_true, y_pred):
        y_pred_positive = K.round(K.clip(y_pred, 0, 1))

        y_positive = K.round(K.clip(y_true, 0, 1))
        y_negative = 1 - y_positive

        TP = K.sum(y_positive * y_pred_positive)
        FP = K.sum(y_negative * y_pred_positive)
        PC = TP/(TP + FP + K.epsilon())
        return PC

    


    def specificity(self,y_true, y_pred):
        y_pred_positive = K.round(K.clip(y_pred, 0, 1))
        y_pred_negative = 1 - y_pred_positive

        y_positive = K.round(K.clip(y_true, 0, 1))

        y_negative = 1 - y_positive

        #TP = K.sum(y_positive * y_pred_positive)
        TN = K.sum(y_negative * y_pred_negative)

        FP = K.sum(y_negative * y_pred_positive)
        #FN = K.sum(y_positive * y_pred_negative)
        SP = TN / (TN + FP + K.epsilon())
        return SP


    def f1_socre(self,y_true, y_pred):
        y_pred_positive = K.round(K.clip(y_pred, 0, 1))
        y_pred_negative = 1 - y_pred_positive

        y_positive = K.round(K.clip(y_true, 0, 1))
        y_negative = 1 - y_positive

        TN = K.sum(y_negative * y_pred_negative)

        FP = K.sum(y_negative * y_pred_positive)
        SE = TN / (TN + FP + K.epsilon())
        TP = K.sum(y_positive * y_pred_positive)
        PC = TP/(TP + FP + K.epsilon())
        F1 = 2 * SE * PC / (SE + PC + K.epsilon())
        return F1

    

    '''def get_edge_points(img):
        """ get edge points of a binary segmentation result"""
        dim = len(img.shape)
        if (dim == 2):
            strt = ndimage.generate_binary_structure(2, 1)
        else:
            strt = ndimage.generate_binary_structure(3, 1)  # 三维结构元素，与中心点相距1个像素点的都是邻域
        ero = ndimage.morphology.binary_erosion(img, strt)
        edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8)
        return edge


    def binary_hausdorff95(s, g, spacing=None):
        """
        get the hausdorff distance between a binary segmentation and the ground truth
        inputs:
            s: a 3D or 2D binary image for segmentation
            g: a 2D or 2D binary image for ground truth
            spacing: a list for image spacing, length should be 3 or 2
        """
        s_edge = get_edge_points(s)
        g_edge = get_edge_points(g)
        image_dim = len(s.shape)
        assert (image_dim == len(g.shape))
        if (spacing == None):
            spacing = [1.0] * image_dim
        else:
            assert (image_dim == len(spacing))
        img = np.zeros_like(s)
        if (image_dim == 2):
            s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
            g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
        elif (image_dim == 3):
            s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
            g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

        dist_list1 = s_dis[g_edge > 0]
        dist_list1 = sorted(dist_list1)
        dist1 = dist_list1[int(len(dist_list1) * 0.95)]
        dist_list2 = g_dis[s_edge > 0]
        dist_list2 = sorted(dist_list2)
        dist2 = dist_list2[int(len(dist_list2) * 0.95)]
        return max(dist1, dist2)


    # 平均表面距离
    def binary_assd(s, g, spacing=None):
        """
        get the average symetric surface distance between a binary segmentation and the ground truth
        inputs:
            s: a 3D or 2D binary image for segmentation
            g: a 2D or 2D binary image for ground truth
            spacing: a list for image spacing, length should be 3 or 2
        """
        s_edge = get_edge_points(s)
        g_edge = get_edge_points(g)
        image_dim = len(s.shape)
        assert (image_dim == len(g.shape))
        if (spacing == None):
            spacing = [1.0] * image_dim
        else:
            assert (image_dim == len(spacing))
        img = np.zeros_like(s)
        if (image_dim == 2):
            s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
            g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
        elif (image_dim == 3):
            s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
            g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

        ns = s_edge.sum()
        ng = g_edge.sum()
        s_dis_g_edge = s_dis * g_edge
        g_dis_s_edge = g_dis * s_edge
        assd = (s_dis_g_edge.sum() + g_dis_s_edge.sum()) / (ns + ng)
        return assd'''

    
