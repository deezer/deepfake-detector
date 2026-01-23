import os
from collections import defaultdict
from glob import glob

import numpy as np
import tensorflow as tf
import torchaudio
import sox

default_loader_params = {
    'sr': 44100, # sampling rate
    'batch_size': 32,
    'repeat': 5, # repeat opened file in next batches for efficiency
    'shuffle': 2, # factor of batch size in buffer
    'seed': 123,
    'audio_slice': 3.0, # in seconds
}

default_augmenter_params = {
    "sr": 44100,
    'fft': {
        'win': 512,
        'hop': 256,
        'n_fft': 1024,
    },
    'normalise_mean': 0.,
    'normalise_std': 1.,
    'hf_cut': 16000,
    'lf_cut': 4000,
    'effects': [],
}

default_adversarial_params = {
    "sr": 44100,
    "adversarial_effects": [],
    "target_audio_slice": 0.78,
    "adversarial_pitch": 2,
    "adversarial_stretch": 0.2, # -> [80%, 120%]
}


class AudioLoader:
    def __init__(self, real_db_path, fake_db_path, params = {}, split_path = None, codec = ''):
        """
        Assumptions:
        - real_db is structured as folder/songs.mp3
        - fake_db is structured as encoder/folder/songs.mp3
        """

        self.params = default_loader_params
        self.params.update(params)

        self.pos_list = glob(os.path.join(real_db_path, '**/*.mp3'))
        self.neg_list = glob(os.path.join(fake_db_path, '**/**/*.mp3'))
        if codec != '':
            self.pos_list = glob(os.path.join(real_db_path, '*.{}'.format(codec) ))
            self.neg_list = glob(os.path.join(fake_db_path, '**/*.{}'.format(codec) ))

        if len(self.pos_list) * len(self.neg_list) == 0:
            raise ValueError("DB path incorrect, no file found: {}, {}".format(real_db_path, fake_db_path))

        self.dict_pos = {}
        set_pos = set()
        for s_path in self.pos_list:
            s_mp3 = s_path.split("/")[-1]
            s_mp3 = s_mp3.split('.')[0] + '.mp3'  # fix for other codec...
            self.dict_pos[s_mp3] = s_path
            set_pos.add(s_mp3)
        self.dict_pos = tf.lookup.StaticHashTable(
                initializer=tf.lookup.KeyValueTensorInitializer(
                    keys=tf.constant(list(self.dict_pos.keys())),
                    values=tf.constant(list(self.dict_pos.values())),
                ),
                default_value = tf.constant("NO-FILE"),
            )

        self.dict_neg = defaultdict(dict)
        set_neg = defaultdict(set)
        for s_path in self.neg_list:
            s_path_split = s_path.split("/")
            s_mp3 = s_path_split[-1]
            s_mp3 = s_mp3.split('.')[0] + '.mp3'  # fix for other codec...
            encoder = s_path_split[-3]
            if codec != '':
                encoder = s_path_split[-2]
            if encoder == 'real':
                continue
            self.dict_neg[encoder + "." + s_mp3] = s_path
            set_neg[encoder].add(s_mp3)

        self.encoders = sorted(list(set_neg.keys()))
        self.n_encoders = len(self.encoders)
        self.dict_encoder = tf.lookup.StaticHashTable(
                initializer=tf.lookup.KeyValueTensorInitializer(
                    keys=tf.constant( list(range(self.n_encoders)) ),
                    values=tf.constant(self.encoders),
                ),
                default_value = tf.constant("NO-ENCODER"),
            )

        self.dict_neg = tf.lookup.StaticHashTable(
                initializer=tf.lookup.KeyValueTensorInitializer(
                    keys=tf.constant(list(self.dict_neg.keys())),
                    values=tf.constant(list(self.dict_neg.values())),
                ),
                default_value = tf.constant("NO-FILE"),
            )

        intersection_mp3 = set_pos
        for encoder in set_neg:
            intersection_mp3 = intersection_mp3 & set_neg[encoder]
        self.all_mp3 = list(intersection_mp3)

        np.random.seed(self.params['seed'])
        tf.random.set_seed(self.params['seed'])
        if not split_path:
            print("\nWarning no splitting file given! Defaulting to random split with seed {}\n".format(self.params['seed']))
            self.split_mp3 = {}
            self.split_mp3['train'], self.split_mp3['validation'], self.split_mp3['test'] = self.create_data_splits(self.all_mp3)
        else:
            print("Opening external split path {}".format(split_path))
            self.split_mp3 = np.load(split_path, allow_pickle=True).item()


    def create_data_splits(self, f_list, val_split = 0.1, test_split = 0.2):
        f_list_ = np.copy(f_list)
        np.random.shuffle(f_list_)
        val_split_index = round(len(f_list_) * (1 - val_split - test_split))
        test_split_index = round(len(f_list_) * (1 - test_split))

        return (f_list_[:val_split_index],
                f_list_[val_split_index:test_split_index],
                f_list_[test_split_index:])


    def gen_labels(self, buffer_label = 10000):
        """ generate fake/true labels and encoder labels """
        y_iterator_1 = tf.data.Dataset.from_tensor_slices(
            tf.convert_to_tensor([0, 1], tf.float32)  # alternate
        ).repeat()
        y_iterator_2 = tf.data.Dataset.from_tensor_slices(
            tf.random.uniform( (buffer_label,), 0, self.n_encoders, tf.int32 )
        ).repeat()
        y_iterator = tf.data.Dataset.zip((y_iterator_1, y_iterator_2))
        return y_iterator


    def gen_encoder_labels(self, encoder):
        """ generate fake/true labels and encoder labels """
        if encoder not in self.encoders:
            raise ValueError("Encoder unavailable:", encoder)

        y_iterator_1 = tf.data.Dataset.from_tensor_slices(
            tf.convert_to_tensor([0, 1], tf.float32)  # alternate
        ).repeat()
        idx_encoder = self.encoders.index(encoder)
        y_iterator_2 = tf.data.Dataset.from_tensor_slices( tf.constant([idx_encoder], tf.int32) ).repeat()
        y_iterator = tf.data.Dataset.zip((y_iterator_1, y_iterator_2))
        return y_iterator


    def get_file_path(self, mp3, labels):
        y, idx_encoder = labels
        encoder = self.dict_encoder.lookup( idx_encoder )
        if y > 0:
            return self.dict_pos.lookup(mp3), labels
        return self.dict_neg.lookup( tf.strings.join((encoder, '.', mp3)) ), labels


    @tf.function(reduce_retracing=True)
    def torch_open_audio(self, f_name, labels):
        def py_open_audio(fpath):
            audio_raw, sr = torchaudio.load(
                fpath.numpy().decode('utf-8'),
                channels_first = False,
                )
            audio_raw = audio_raw.numpy()
            # Handle mono tracks: duplicate to stereo
            # audio_raw shape is (time, channels)
            if audio_raw.shape[-1] == 1:
                # Duplicate mono channel to make stereo
                audio_raw = np.repeat(audio_raw, 2, axis=-1)
            return audio_raw

        audio = tf.py_function(
            py_open_audio,
            [ f_name ],
            tf.float32,
        )

        return audio, labels


    @tf.function
    def get_pair_audio(self, mp3, labels):
        y, idx_encoder = labels
        real_path, _ = self.get_file_path(mp3, (tf.ones(1), idx_encoder) )
        fake_path, _ = self.get_file_path(mp3, (tf.zeros(1), idx_encoder) )

        real_audio, _ = self.torch_open_audio(real_path, labels)
        fake_audio, _ = self.torch_open_audio(fake_path, labels)
        min_shape = tf.minimum(tf.shape(real_audio)[0], tf.shape(fake_audio)[0])
        fake_audio = tf.slice( fake_audio, (0, 0), (min_shape, 2) )
        real_audio = tf.slice( real_audio, (0, 0), (min_shape, 2) )
        return tf.stack((real_audio, fake_audio), 0)


    @tf.function
    def pair_mixing(self, n_bins, audios):
        labels = tf.cast(tf.linspace(0, 1, n_bins), tf.float32)
        mixing = tf.stack((labels, 1 - labels), 1)

        return tf.tensordot(mixing, audios, 1), labels


    @tf.function
    def slice_audio(self, x, labels):
        """ assumes channels last """
        target_len = int(self.params['audio_slice'] * self.params['sr'])
        max_offset = tf.shape(x)[-2] - target_len
        offset = tf.random.uniform((), maxval = max_offset + 1, dtype = tf.int32)
        return tf.slice(x, (offset, 0), (target_len, 2)), labels


    @tf.function
    def patch_spec(self, x, labels):
        """ cut into a spectrogram """
        x_shape = tf.shape(x)
        max_offset_t = x_shape[0] - self.params['patch_size_t']
        max_offset_f = x_shape[1] - self.params['patch_size_f']
        min_offset_f = 0
        if 'patch_f_min_threshold' in self.params:
            min_offset_f = self.params['patch_f_min_threshold']
        if 'patch_f_max_threshold' in self.params:
            max_offset_f = self.params['patch_f_max_threshold'] - self.params['patch_size_f']
        offset_t = tf.random.uniform((), maxval = max_offset_t + 1, dtype = tf.int32)
        offset_f = tf.random.uniform((), minval=min_offset_f,
                        maxval = max_offset_f + 1, dtype = tf.int32)
        return tf.slice(x, (offset_t, offset_f, 0), (self.params['patch_size_t'], self.params['patch_size_f'], -1)), labels


    @tf.function
    def patch_batch_spec(self, x, labels):
        """ same but random batch """
        x_shape = tf.shape(x)
        max_offset_t = x_shape[1] - self.params['patch_size_t']
        patch_size_f = tf.random.uniform((), minval = self.params['patch_size_f_min'],
                                            maxval = self.params['patch_size_f'] + 1,
                                            dtype = tf.int32)

        max_offset_f = x_shape[2] - patch_size_f
        offset_t = tf.random.uniform((), maxval = max_offset_t + 1, dtype = tf.int32)
        offset_f = tf.random.uniform((), maxval = max_offset_f + 1, dtype = tf.int32)
        return ( tf.slice(x,
                    (0, offset_t, offset_f, 0),
                    (-1, self.params['patch_size_t'], patch_size_f, -1) ),
                labels )


    def create_tf_iterator(self, mode = 'train', augmenter = None, encoder=None):
        iterator = tf.data.Dataset.from_tensor_slices(self.split_mp3[mode])
        iterator = iterator.repeat()
        if not encoder:
            y_iterator = self.gen_labels()
        else:
            y_iterator = self.gen_encoder_labels(encoder)

        iterator = tf.data.Dataset.zip(( iterator, y_iterator ))
        iterator = iterator.map(self.get_file_path)

        iterator = iterator.map(self.torch_open_audio, num_parallel_calls=tf.data.AUTOTUNE)
        iterator = iterator.flat_map(lambda *x:
                tf.data.Dataset.from_tensors(x).repeat(self.params['repeat']) )
        iterator = iterator.map( self.slice_audio )
        if augmenter:
            iterator = iterator.map( lambda x, y: (augmenter.transform(x), y) )
        iterator = iterator.shuffle(self.params['batch_size'] * self.params['repeat'] * self.params['shuffle'])

        iterator = iterator.batch(
                self.params['batch_size'],
                num_parallel_calls=tf.data.AUTOTUNE,
                )
        iterator = iterator.prefetch( tf.data.AUTOTUNE )

        return iterator


    def create_fast_eval_iterator(self, encoder = 'real', mode = 'test',
                augmenter = None, adversarial = None ):
        """ batch adversarial """

        iterator = tf.data.Dataset.from_tensor_slices(self.split_mp3[mode])
        if encoder == 'real':
            y_iterator_1 = tf.data.Dataset.from_tensor_slices( tf.ones((1,), tf.float32) ).repeat()
            y_iterator_2 = tf.data.Dataset.from_tensor_slices( tf.constant([-1], tf.int32) ).repeat()
        else:
            if encoder not in self.encoders:
                raise ValueError("Encoder unavailable:", encoder)
            idx_encoder = self.encoders.index(encoder)
            y_iterator_1 = tf.data.Dataset.from_tensor_slices( tf.zeros((1,), tf.float32) ).repeat()
            y_iterator_2 = tf.data.Dataset.from_tensor_slices( tf.constant([idx_encoder], tf.int32) ).repeat()
        y_iterator = tf.data.Dataset.zip((y_iterator_1, y_iterator_2))

        iterator = tf.data.Dataset.zip(( iterator, y_iterator ))
        iterator = iterator.map(self.get_file_path)
        iterator = iterator.map(self.torch_open_audio,  # self.open_audio
                                num_parallel_calls=tf.data.AUTOTUNE,
                                )
        iterator = iterator.flat_map(lambda *x:
                tf.data.Dataset.from_tensors(x).repeat(self.params['repeat']) )
        iterator = iterator.map( self.slice_audio )
        iterator = iterator.batch(
                self.params['batch_size'],
                num_parallel_calls=tf.data.AUTOTUNE,
                )
        if adversarial:
            iterator = iterator.map( lambda x, y: (adversarial.batch_transform(x), y),
                            num_parallel_calls=tf.data.AUTOTUNE )
        iterator = iterator.unbatch()
        if augmenter:
            iterator = iterator.map( lambda x, y: (augmenter.transform(x), y) )
        iterator = iterator.batch(
                self.params['batch_size'],
                num_parallel_calls=tf.data.AUTOTUNE,
                )
        iterator = iterator.prefetch( tf.data.AUTOTUNE )

        return iterator



    def create_calibration_iterator(self, mode = 'test', n_bins=10, augmenter = None):
        """ only used for computing the model calibration, tests different audio mix """
        iterator = tf.data.Dataset.from_tensor_slices(self.split_mp3[mode])
        y_iterator = self.gen_labels()  # but ignore y[0]

        iterator = tf.data.Dataset.zip(( iterator, y_iterator ))
        iterator = iterator.map(self.get_pair_audio,
                                num_parallel_calls=tf.data.AUTOTUNE,
                                )
        iterator = iterator.map(lambda x: self.pair_mixing(n_bins, x) )
        iterator = iterator.unbatch()
        iterator = iterator.map( self.slice_audio )
        if augmenter:
            iterator = iterator.map( lambda x, y: (augmenter.transform(x), y) )
        iterator = iterator.batch(
                self.params['batch_size'],
                num_parallel_calls=tf.data.AUTOTUNE,
                )
        iterator = iterator.prefetch( tf.data.AUTOTUNE )

        return iterator



    def create_patch_iterator(self, augmenter, mode = 'train'):
        ''' Train the patch model for spectrogram interpretability '''
        iterator = tf.data.Dataset.from_tensor_slices(self.split_mp3[mode])
        iterator = iterator.repeat()
        y_iterator = self.gen_labels()

        iterator = tf.data.Dataset.zip(( iterator, y_iterator ))
        iterator = iterator.map(self.get_file_path)

        iterator = iterator.map(self.torch_open_audio,  # self.open_audio
                                num_parallel_calls=tf.data.AUTOTUNE,
                                )
        iterator = iterator.map( self.slice_audio ) # accelerate before stft
        iterator = iterator.map( lambda x, y: (augmenter.transform(x), y) )
        iterator = iterator.flat_map(lambda *x:
                tf.data.Dataset.from_tensors(x).repeat(self.params['repeat']) )
        iterator = iterator.map( self.patch_spec )
        iterator = iterator.shuffle(self.params['batch_size'] * self.params['repeat'] * self.params['shuffle'])
        iterator = iterator.batch(
                self.params['batch_size'],
                num_parallel_calls=tf.data.AUTOTUNE,
                )
        iterator = iterator.prefetch( tf.data.AUTOTUNE )

        return iterator





class Augmenter:
    def __init__(self, params = {}):
        self.params = default_augmenter_params
        self.params.update(params)

        self.normaliser_mean = tf.constant(self.params["normalise_mean"], tf.float32)
        self.normaliser_std = tf.constant(self.params["normalise_std"], tf.float32)


    def __bool__(self):
        return True

    @staticmethod
    def switch_channels(x):
        return tf.transpose(x, [1, 0])


    @tf.function
    def stft(self, x, mode = 'complex'):
        if x.shape[-1] == 2: # stereo
            x = self.switch_channels(x)

        complex_x = tf.signal.stft(
                x,
                self.params['fft']['win'],
                self.params['fft']['hop'],  # pas temporel de hop/sr
                fft_length = self.params['fft']['n_fft'],
            )
        if mode == 'magnitude':
            return tf.expand_dims(tf.abs(complex_x), -1)

        elif mode == 'power':
            return tf.expand_dims(tf.square(tf.abs(complex_x)), -1)

        elif mode == 'dB':
            return tf.expand_dims( tf.math.log( tf.clip_by_value(
                        tf.square(tf.abs(complex_x)),
                        1e-10,
                        1e6, )
                    ) / tf.math.log( tf.constant(10., dtype=tf.float32)), -1)

        elif mode == 'polar':
            return tf.concat((tf.abs(complex_x), tf.math.angle(complex_x)), -1)

        elif mode == "pure_phase":
            angle = tf.math.angle(complex_x)  # non continu
            print("angle", angle.shape)
            angle_cossin = tf.stack((tf.math.cos(angle), tf.math.sin(angle)), -1)
            return angle_cossin

        else:
            return tf.stack((tf.math.real(complex_x), tf.math.imag(complex_x)), -1)

    @tf.function
    def normaliser(self, x):
        return (x - self.normaliser_mean) / self.normaliser_std

    def random_mono(self, x):
        if x.shape[0] == 2: # stereo channel first
            x = self.switch_channels(x)

        factor = tf.random.uniform((), minval = 0.01, maxval=0.99, dtype = tf.float32)
        return factor * x[:,0] + (1 - factor) * x[:,1]

    def random_affine(self, x):
        factor = tf.random.uniform((), minval = 0.5, maxval=1.0, dtype = tf.float32)
        return factor * x

    def slice_hf(self, x):
        factor = tf.cast((self.params["hf_cut"] * 2 / self.params["sr"]) * tf.cast(tf.shape(x)[1], tf.float32), tf.int32)
        return tf.slice(x, (0,0,0), (-1,factor,-1))

    def slice_lf(self, x,):
        factor = tf.cast((self.params["lf_cut"] * 2 / self.params["sr"]) * tf.cast(tf.shape(x)[1], tf.float32), tf.int32)
        return tf.slice(x, (0,factor,0), (-1,-1,-1))

    def add_noise(self, x):
        noise = tf.random.normal(tf.shape(x), stddev=1e-2, dtype=tf.float32)
        return x + noise

    @tf.function
    def transform(self, x, skip_normalise = False):
        y = x
        for effect in self.params["effects"]:
            if effect == 'stft_db':
                y = self.stft(y, 'dB')
            elif effect == 'stft_mag':
                y = self.stft(y, 'magnitude')
            elif effect == 'stft_complex':
                y = self.stft(y, 'complex')
            elif effect == 'stft_polar':
                y1 = self.stft(y, 'pure_phase')
                y2 = self.normaliser( self.stft(y, 'dB') )
                y = tf.concat((y1, y2), -1)
            elif effect == 'stft_phase':
                y = self.stft(y, 'pure_phase')
            elif effect == 'normalise' and not skip_normalise:
                y = self.normaliser(y)
            elif effect == 'mono':
                y = self.random_mono(y)
            elif effect == 'affine':
                y = self.random_affine(y)
            elif effect == 'slice_hf':
                y = self.slice_hf(y)
            elif effect == 'slice_lf':
                y = self.slice_lf(y)
            elif effect == "noise":
                y = self.add_noise(y)

        return y



class EvalAugmenter(Augmenter):
    def __init__(self, params = {}):
        super().__init__(params)


    def random_mono(self, x):
        if x.shape[0] == 2: # stereo channel first
            x = self.switch_channels(x)
        return 0.5 * x[:,0] + 0.5 * x[:,1]

    def random_affine(self, x):
        return x  # disable



class AdversarialAugmenter:
    def __init__(self, params = {}):
        self.params = default_adversarial_params
        self.params.update(params)

        self.target_length = int(self.params["target_audio_slice"] * self.params["sr"])


    def __bool__(self):
        return True


    def create_augmentation(self):
        tfm = sox.Transformer()

        for effect in self.params["adversarial_effects"]:
            if effect == "pitch":
                sign = 2 * np.random.randint(0, 2) - 1
                fact = np.random.randint(1, self.params["adversarial_pitch"]+1)
                factor = fact * sign
                tfm.pitch(n_semitones=factor)

            if effect == "stretch":
                factor = np.random.uniform(
                        1 - self.params["adversarial_stretch"],
                        1 + self.params["adversarial_stretch"], )
                if 0.9 <= factor <= 1.1:
                    tfm.stretch(factor=factor)
                else:
                    tfm.tempo(factor=factor, audio_type="m")


            if effect == "reverb":
                tfm.reverb(
                    reverberance=np.random.uniform(20, 80),
                    high_freq_damping=np.random.uniform(20, 80),
                    room_scale=np.random.choice([100, np.random.uniform(50, 100)]),
                    stereo_depth=np.random.choice([100, np.random.uniform(50, 100)]),
                    pre_delay=np.random.choice([0, np.random.uniform(0, 5)]),
                    wet_gain=np.random.uniform(-3, 3),
                    wet_only=False,
                )

            if effect == "eq":
                if not random.randrange(3):
                    tfm.bandreject(
                        frequency=np.random.uniform(80, 8000),
                        width_q=np.random.uniform(2, 8),
                    )
                if not random.randrange(3):
                    tfm.bass(
                        gain_db=np.random.uniform(-20, 20),
                        frequency=np.random.uniform(60, 140),
                        slope=np.random.uniform(0.3, 1.0),
                    )
                if not random.randrange(3):
                    tfm.treble(
                        gain_db=np.random.uniform(-20, 20),
                        frequency=np.random.uniform(1000, 5000),
                        slope=np.random.uniform(0.3, 1.0),
                    )
                if not random.randrange(3):
                    tfm.equalizer(
                        frequency=np.random.uniform(80, 8000),
                        width_q=np.random.uniform(0.5, 4),
                        gain_db=np.random.uniform(-5, 5),
                    )
                if not random.randrange(3):
                    tfm.highpass(
                        frequency=np.random.uniform(20, 500),
                        width_q=np.random.uniform(0.5, 4),
                        n_poles=2,
                    )
                if not random.randrange(3):
                    tfm.lowpass(
                        frequency=np.random.uniform(500, 8000),
                        width_q=np.random.uniform(0.5, 4),
                        n_poles=2,
                    )

        return tfm

    def py_apply_augmentation(self, audio):
        tfm = self.create_augmentation()
        return tfm.build_array(input_array=audio.numpy(),
                                       sample_rate_in=self.params["sr"])

    @tf.function
    def transform(self, x):
        x_shape = tf.shape(x)
        x_ = tf.py_function(self.py_apply_augmentation, [x], tf.float32)
        return tf.slice( x_, (0, 0), x_shape)

    @tf.function
    def batch_transform(self, X):
        X_shape = tf.shape(X)
        X_trick_sox = tf.reshape( tf.transpose(X, [1, 0, 2]), (X_shape[1], -1) )
        X_ = tf.py_function(self.py_apply_augmentation, [X_trick_sox], tf.float32)
        X_ = tf.transpose(
                tf.reshape( X_, (-1, X_shape[0], X_shape[2])  ),
                [1, 0, 2] )
        # X_ = tf.slice( X_, (0, 0, 0), X_shape)
        return X_

