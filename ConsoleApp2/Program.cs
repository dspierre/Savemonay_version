//using System.Numerics.Tensors;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
namespace ConsoleAppOnnx
{
    class Program
    {
        static void Main(string[] args)
        {
            var opts = new Microsoft.AspNetCore.Builder.SessionOptions();
            string model_path = @"C:\xadupre\microsoft_xadupre\sklearn-onnx\tests\tests_dump\SklearnPipelineColumnTransformerPipelinerOptions2.model.onnx";
            var session = new InferenceSession(model_path, opts);
            var dims = new int[] { 1, 111 };
            var t = new DenseTensor<float>(dims);
            t.Fill(0);
            //object NamedOnnxValue = null;
            var tensor = NamedOnnxValue.CreateFromTensor("name1", t);
            //object[] ps = new[] { tensor };
            //using var outputs = session.Run(ps);
            /*foreach (var o in outputs)
            {
                DenseTensor<float> to = o.AsTensor<float>().ToDenseTensor();
                var values = new float[to.Length];
                to.Buffer.CopyTo(values);
            }*/
            using (var outputs = session.Run(new[] { tensor }))
            {
                foreach (var o in outputs)
                {
                    DenseTensor<float> to = o.AsTensor<float>().ToDenseTensor();
                    var values = new float[to.Length];
                    to.Buffer.CopyTo(values);
                }
            }
        }
        private class InferenceSession
        {
            private string model_path;
            private Microsoft.AspNetCore.Builder.SessionOptions opts;

            public InferenceSession(string model_path, Microsoft.AspNetCore.Builder.SessionOptions opts)
            {
                this.model_path = model_path;
                this.opts = opts;
            }
            internal System.IDisposable Run(object[] ps)
            {
                throw new System.NotImplementedException();
            }
    }
    }
}
