"""EXL3 logical storage, grouped projections, and malformed header rejection."""
import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from fastllm_pytools.weight_formats import Tensor, _exl3_inventory

def projection(prefix, words=48):
    return [Tensor(prefix+'.trellis',(8,8,words),128*words,'I16'),
            Tensor(prefix+'.suh',(128,),256,'F16'),Tensor(prefix+'.svh',(128,),256,'F16'),
            Tensor(prefix+'.mul1',(),4,'I32')]

class Exl3InventoryTest(unittest.TestCase):
    def test_native_bytes_and_all_bitrates(self):
        for words in (16,24,32,40,48,56,64,80,96,112,128):
            with self.subTest(words=words):
                tensors=_exl3_inventory(projection('layers.0.ffn.experts.0.w1',words))
                self.assertEqual(tensors,[Tensor('layers.0.ffn.experts.0.w1.weight',(128,128),128*words+512,'EXL3')])
    def test_storage_capabilities_and_calibration(self):
        for words in (16,24,32,40,48,56,64,80,96,112,128):
            tensor = _exl3_inventory(projection('head',words))[0]
            self.assertTrue(tensor.preserve_source)
            self.assertFalse(tensor.supports_disk)
            self.assertEqual(tensor.calibration_override,-words//8)
        gguf = Tensor('head.weight',(128,128),6144,'gguf:12')
        dense = Tensor('head.weight',(128,128),32768,'BF16')
        self.assertTrue(gguf.preserve_source)
        self.assertTrue(gguf.supports_disk)
        self.assertFalse(dense.preserve_source)
        self.assertIsNone(dense.calibration_override)

    def test_grouped_projection_and_unquantized(self):
        normal=Tensor('embed.weight',(16,128),4096,'BF16')
        tensors=projection('layers.0.attn.wq_a')+[normal]
        for i in range(8):tensors+=projection('layers.0.attn.wo_a.slice.'+str(i))
        actual={t.name:t for t in _exl3_inventory(tensors)}
        self.assertEqual(actual['embed.weight'],normal)
        self.assertEqual(actual['layers.0.attn.wo_a.weight'],Tensor('layers.0.attn.wo_a.weight',(8,128,128),8*(128*48+512),'EXL3'))
        self.assertEqual(len(actual),3)
    def test_grouped_mixed_bits_keeps_float_fallback(self):
        tensors=projection('head')+projection('layers.0.attn.wo_a.slice.0',48)+projection('layers.0.attn.wo_a.slice.1',64)
        actual={t.name:t for t in _exl3_inventory(tensors)}
        self.assertEqual(actual['layers.0.attn.wo_a.weight'],Tensor('layers.0.attn.wo_a.weight',(2,128,128),2*128*128*2,'BF16'))
    def test_missing_scale_or_codebook(self):
        for missing in range(1,4):
            tensors=projection('head');del tensors[missing]
            with self.assertRaisesRegex(ValueError,'incomplete'): _exl3_inventory(tensors)
    def test_malformed_dimensions_or_bytes(self):
        for shape,size in [((7,8,48),6144),((8,8,49),6272),((8,8,48),1)]:
            tensors=projection('head');tensors[0]=Tensor('head.trellis',shape,size,'I16')
            with self.assertRaisesRegex(ValueError,'invalid'): _exl3_inventory(tensors)
    def test_missing_group_slice(self):
        tensors=projection('head')+projection('layers.0.attn.wo_a.slice.1')
        with self.assertRaisesRegex(ValueError,'grouped'): _exl3_inventory(tensors)

if __name__=='__main__':unittest.main()
